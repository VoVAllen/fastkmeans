from __future__ import annotations

import time

import torch
import numpy as np

try:
    from fastkmeans.triton_kernels import triton_kmeans

    HAS_TRITON = True
except ImportError:
    triton_kmeans = None
    HAS_TRITON = False


def _get_device(preset: str | int | torch.device | None = None):
    if isinstance(preset, torch.device):
        return preset
    if isinstance(preset, str):
        return torch.device(preset)
    if torch.cuda.is_available():  # cuda currently handles both AMD and NVIDIA GPUs
        return torch.device(f"cuda:{preset if isinstance(preset, int) and preset < torch.cuda.device_count() else 0}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(f"xpu:{preset if isinstance(preset, int) and preset < torch.xpu.device_count() else 0}")
    return torch.device("cpu")


def _is_bfloat16_supported(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    elif device.type == "xpu" and hasattr(torch.xpu, "is_bf16_supported"):
        return torch.xpu.is_bf16_supported()
    else:
        return False


@torch.inference_mode()
def _kmeans_torch_double_chunked(
    data: torch.Tensor,
    data_norms: torch.Tensor,
    k: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_iters: int = 25,
    tol: float = 1e-8,
    chunk_size_data: int = 50_000,
    chunk_size_centroids: int = 10_000,
    max_points_per_centroid: int = 256,
    verbose: bool = False,
    use_triton: bool | None = None,
    metric: str = "euclidean",
    epsilon: float = 1e-8,
):
    """
    An efficient kmeans implementation that minimises OOM risks on modern hardware by using conversative double chunking.

    Parameters
    ----------
    data : torch.Tensor
        Input data tensor of shape (n_samples, n_features). If metric is "cosine",
        this data is L2 normalized internally at the beginning of the function.
    data_norms : torch.Tensor
        Tensor containing the squared L2 norms of each data point. Used only if metric is "euclidean".
        Ignored if metric is "cosine".
    k : int
        Number of clusters.
    device : torch.device
        The device to perform calculations on.
    dtype : torch.dtype | None, default=None
        The data type to use for calculations. If None, defaults to float16 on GPU, float32 on CPU.
    max_iters : int, default=25
        Maximum number of iterations.
    tol : float, default=1e-8
        Tolerance for centroid shift to declare convergence.
    chunk_size_data : int, default=50_000
        Number of data points to process in each chunk.
    chunk_size_centroids : int, default=10_000
        Number of centroids to process in each chunk during distance calculation.
    max_points_per_centroid : int, default=256
        If n_samples > k * max_points_per_centroid, data is randomly subsampled.
    verbose : bool, default=False
        Whether to print iteration information.
    use_triton : bool | None, default=None
        Whether to use Triton kernel for distance calculations.
        If `metric` is "cosine", the PyTorch path is taken as Triton is only explicitly called for "euclidean"
        within this function's logic.
        If None, attempts to use if available and supported for "euclidean" metric.
    metric : str, default="euclidean"
        The distance metric to use. Can be "euclidean" or "cosine".
        - "euclidean": Standard Euclidean distance. `data_norms` is utilized.
        - "cosine": Cosine similarity based distance (1 - cosine_similarity). Input `data` and initial
          centroids are L2 normalized. Subsequent centroid updates are also normalized. `data_norms` is ignored.
    epsilon : float, default=1e-8
        Small value to add to the denominator during L2 normalization to prevent division by zero,
        especially for zero-norm vectors when `metric` is "cosine".

    Returns
    -------
    centroids_cpu : torch.Tensor, shape (k, n_features), float32
    labels_cpu    : torch.Tensor, shape (n_samples_used,), long
        Where n_samples_used can be smaller than the original if subsampling occurred.
    """

    if use_triton:
        if not HAS_TRITON:
            raise ImportError("Triton is not available. Please install Triton and try again.")

    if dtype is None:
        dtype = torch.float16 if device.type in ["cuda", "xpu"] else torch.float32

    n_samples_original, n_features = data.shape
    n_samples = n_samples_original

    if max_points_per_centroid is not None and n_samples > k * max_points_per_centroid:
        target_n_samples = k * max_points_per_centroid
        perm = torch.randperm(n_samples, device=data.device)
        indices = perm[:target_n_samples]
        data = data[indices]
        # data_norms is not used if metric is cosine, but we need to slice it anyway
        data_norms = data_norms[indices]
        n_samples = target_n_samples
        del perm, indices

    if n_samples < k:
        raise ValueError(f"Number of training points ({n_samples}) is less than k ({k}).")

    # Epsilon is now passed as a parameter
    if metric == "cosine":
        data_norm = torch.linalg.norm(data, dim=1, keepdim=True)
        data = data / (data_norm + epsilon) # Use passed epsilon

    # centroid init -- random is the only supported init
    rand_indices = torch.randperm(n_samples)[:k]
    centroids = data[rand_indices].clone().to(device=device, dtype=dtype)
    if metric == "cosine":
        centroid_norm = torch.linalg.norm(centroids, dim=1, keepdim=True)
        centroids = centroids / (centroid_norm + epsilon) # Use passed epsilon
    prev_centroids = centroids.clone()

    labels = torch.empty(n_samples, dtype=torch.int64, device="cpu")  # Keep labels on CPU

    for iteration in range(max_iters):
        iteration_start_time = time.time()

        if metric == "euclidean":
            centroid_norms = (centroids**2).sum(dim=1)
        cluster_sums = torch.zeros((k, n_features), device=device, dtype=torch.float32)
        cluster_counts = torch.zeros((k,), device=device, dtype=torch.float32)

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + chunk_size_data, n_samples)

            data_chunk = data[start_idx:end_idx].to(device=device, dtype=dtype, non_blocking=True)
            if metric == "euclidean":
                data_chunk_norms = data_norms[start_idx:end_idx].to(device=device, dtype=dtype, non_blocking=True)
            batch_size = data_chunk.size(0)
            best_ids = torch.zeros((batch_size,), device=device, dtype=torch.int64)

            if use_triton and metric == "euclidean": # Triton kernel only supports euclidean
                triton_kmeans(
                    data_chunk=data_chunk,
                    data_chunk_norms=data_chunk_norms,
                    centroids=centroids,
                    centroids_sqnorm=centroid_norms,
                    best_ids=best_ids,
                )
            else:
                best_dist = torch.full((batch_size,), float("inf"), device=device, dtype=dtype)
                c_start = 0
                while c_start < k:
                    c_end = min(c_start + chunk_size_centroids, k)
                    centroid_chunk = centroids[c_start:c_end]
                    
                    if metric == "euclidean":
                        centroid_chunk_norms = centroid_norms[c_start:c_end]
                        dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                        dist_chunk = dist_chunk.addmm_(data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0)
                    elif metric == "cosine":
                        # For cosine, best is highest dot product, so we minimize 1 - dot_product
                        dist_chunk = 1 - torch.matmul(data_chunk, centroid_chunk.t())


                    local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                    improved_mask = local_min_vals < best_dist
                    best_dist[improved_mask] = local_min_vals[improved_mask]
                    best_ids[improved_mask] = c_start + local_min_ids[improved_mask]

                    c_start = c_end

            cluster_sums.index_add_(0, best_ids, data_chunk.float()) # use original data for sums
            cluster_counts.index_add_(0, best_ids, torch.ones_like(best_ids, device=device, dtype=torch.float32))

            labels[start_idx:end_idx] = best_ids.to("cpu", non_blocking=True)
            start_idx = end_idx

        new_centroids = torch.zeros_like(centroids, device=device, dtype=dtype)
        non_empty = cluster_counts > 0
        new_centroids[non_empty] = (cluster_sums[non_empty] / cluster_counts[non_empty].unsqueeze(1)).to(dtype=dtype)
        
        if metric == "cosine":
            if new_centroids[non_empty].numel() > 0: # Ensure there are non-empty centroids to normalize
                new_centroids_norm = torch.linalg.norm(new_centroids[non_empty], dim=1, keepdim=True)
                new_centroids[non_empty] = new_centroids[non_empty] / (new_centroids_norm + epsilon) # Use passed epsilon


        empty_ids = (~non_empty).nonzero(as_tuple=True)[0]
        if len(empty_ids) > 0:
            reinit_indices = torch.randint(0, n_samples, (len(empty_ids),), device="cpu")
            random_data = data[reinit_indices].to(device=device, dtype=dtype, non_blocking=True)
            if metric == "cosine": # ensure reinitialized centroids are normalized
                if random_data.numel() > 0: # Ensure there is data to normalize
                    random_data_norm = torch.linalg.norm(random_data, dim=1, keepdim=True)
                    random_data = random_data / (random_data_norm + epsilon) # Use passed epsilon
            new_centroids[empty_ids] = random_data

        if metric == "euclidean":
            shift = torch.norm(new_centroids - prev_centroids.to(new_centroids.device), dim=1).sum().item()
        elif metric == "cosine":
            # For cosine, shift is 1 - dot_product(new, prev)
            # Ensure prev_centroids is on the same device and normalized for cosine
            prev_centroids_device = prev_centroids.to(new_centroids.device)
            # No need to re-normalize prev_centroids as they were normalized in previous iteration or init
            shift = (1 - (new_centroids * prev_centroids_device).sum(dim=1)).sum().item()


        centroids = new_centroids

        prev_centroids = centroids.clone()

        iteration_time = time.time() - iteration_start_time
        if verbose:
            print(
                f"Iteration {iteration + 1}/{max_iters} took {iteration_time:.4f}s, total time: {time.time() - iteration_start_time + iteration_time:.4f}s, shift: {shift:.6f}"
            )

        if shift < tol:
            if verbose:
                print(f"Converged after {iteration + 1} iterations (shift: {shift:.6f} < tol: {tol})")
            break

    centroids_cpu = centroids.to("cpu", dtype=torch.float32)
    return centroids_cpu, labels


class FastKMeans:
    """
    A drop-in replacement for Faiss's Kmeans API, implemented with PyTorch
    double-chunked KMeans under the hood.

    Parameters
    ----------
    d  : int
        Dimensionality of the input features (n_features).
    k  : int
        Number of clusters.
    niter : int, default=25
        Maximum number of iterations for the KMeans algorithm.
    tol : float, default=1e-8
        Stopping threshold for centroid movement. If the sum of squared differences
        between old and new centroids (or sum of `1 - cosine_similarity` for cosine metric)
        is less than this value, the algorithm converges.
    gpu : bool, default=True
        Deprecated. Use `device` parameter instead. If False, forces CPU.
        If True (default), uses available GPU or falls back to CPU.
    seed : int, default=0
        Random seed for centroid initialization and (if needed) subsampling,
        ensuring reproducibility.
    max_points_per_centroid : int, optional, default=1_000_000_000
        If the number of samples `n_samples` exceeds `k * max_points_per_centroid`,
        the input data will be randomly subsampled to `k * max_points_per_centroid`
        points before clustering. This helps manage memory for very large datasets.
    chunk_size_data : int, default=51_200
        The number of data points processed in each chunk during the assignment step.
        Helps manage memory usage.
    chunk_size_centroids : int, default=10_240
        The number of centroids processed in each chunk during the assignment step
        (for distance calculations). Helps manage memory usage.
    device : str | int | torch.device | None, default=None
        The device to use for computation (e.g., "cpu", "cuda", "cuda:0",
        `torch.device("cuda:1")`). If None, behavior is determined by `gpu` flag
        or defaults to the best available device.
    dtype : torch.dtype, default=None
        The PyTorch data type to use for computations (e.g., `torch.float32`, `torch.float16`).
        If None, defaults to `torch.float16` on GPU if supported, else `torch.float32`.
    pin_gpu_memory : bool, default=True
        If True and using a CUDA device, pins CPU memory for `data_torch` and
        `data_norms_torch` (if applicable for the metric) for faster CPU to GPU transfers.
    verbose : bool, default=False
        If True, prints information about iteration progress and convergence.
    nredo : int, default=1
        For Faiss compatibility only. This implementation does not support multiple redos (`nredo > 1`).
    use_triton : bool | None, default=None
       If True, attempts to use the Triton backend for accelerated distance calculations
       on compatible GPUs. If False, uses PyTorch operations. If None, it's auto-detected
       based on GPU capability and Triton availability.
       When `metric` is "cosine" in `_kmeans_torch_double_chunked` (the internal training loop),
       the PyTorch path is always used because the Triton kernel is only explicitly invoked for "euclidean".
       However, the main `triton_kmeans` function itself (in `triton_kernels.py`) has been updated
       to support "cosine" and can be used directly if needed elsewhere (e.g. in `predict` if it were to use Triton for cosine).
    metric : str, default="euclidean"
        The distance metric to use for clustering.
        - "euclidean": Standard L2 Euclidean distance.
        - "cosine": Cosine similarity based distance (clusters are formed based on the angle
          between vectors). Input data will be L2 normalized internally during `train` and `predict`,
          and centroids will also be L2 normalized. This is suitable for spherical KMeans.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        gpu: bool = True,
        seed: int = 0,
        max_points_per_centroid: int = 1_000_000_000,
        chunk_size_data: int = 51_200,
        chunk_size_centroids: int = 10_240,
        device: str | int | torch.device | None = None,
        dtype: torch.dtype = None,
        pin_gpu_memory: bool = True,
        verbose: bool = False,
        nredo: int = 1,  # for compatibility only
        use_triton: bool | None = None,
        metric: str = "euclidean",
        epsilon: float = 1e-8,
    ):
        self.d = d
        self.k = k
        self.niter = niter
        self.tol = tol
        self.seed = seed
        self.max_points_per_centroid = max_points_per_centroid
        self.chunk_size_data = chunk_size_data
        self.chunk_size_centroids = chunk_size_centroids
        self.device = _get_device("cpu" if gpu is False else device)
        self.centroids = None
        self.dtype = dtype
        self.pin_gpu_memory = pin_gpu_memory
        self.verbose = verbose
        if use_triton is None:
            # assume triton kernel is supported if GPU supports bfloat16
            use_triton = HAS_TRITON and _is_bfloat16_supported(self.device)
        if use_triton and not HAS_TRITON:
            raise ValueError("Triton is not available. Please install Triton and try again.")
        self.use_triton = use_triton
        if nredo != 1:
            raise ValueError("nredo must be 1, redos not currently supported")
        self.metric = metric
        self.epsilon = epsilon

    def train(self, data: np.ndarray):
        """
        Trains (fits) the KMeans model on the given data and sets `self.centroids`. Designed to mimic faiss's `train()` method.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, d), float32
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        # Move data to PyTorch CPU Tensor
        data_torch = torch.from_numpy(data)
        
        if self.metric == "cosine":
            data_norm = torch.linalg.norm(data_torch, dim=1, keepdim=True)
            data_torch = data_torch / (data_norm + self.epsilon) # Use self.epsilon
            # data_norms_torch is not used for cosine, initialize to empty or zeros
            data_norms_torch = torch.empty(data_torch.shape[0], device=data_torch.device, dtype=data_torch.dtype)
        elif self.metric == "euclidean":
            data_norms_torch = (data_torch**2).sum(dim=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


        device = _get_device(self.device)
        if device == "cuda" and self.pin_gpu_memory: # should be device.type
            data_torch = data_torch.pin_memory()
            if self.metric == "euclidean": # only pin if it's used
                data_norms_torch = data_norms_torch.pin_memory()

        centroids, _ = _kmeans_torch_double_chunked(
            data_torch,
            data_norms_torch, # This will be ignored if metric is cosine inside the function
            k=self.k,
            max_iters=self.niter,
            tol=self.tol,
            device=device,
            dtype=self.dtype,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
            max_points_per_centroid=self.max_points_per_centroid,
            verbose=self.verbose,
            use_triton=self.use_triton,
            metric=self.metric,
            epsilon=self.epsilon, # Pass self.epsilon
        )
        self.centroids = centroids.numpy()

    def fit(self, data: np.ndarray):
        """
        Same as train(), included for interface similarity with scikit-learn's `fit()`.
        """
        self.train(data)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Assigns each data point to the nearest centroid for even more compatibility with scikit-learn's `predict()`, which is what cool libraries do.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,), int64
        """
        if self.centroids is None:
            raise RuntimeError("Must call train() or fit() before predict().")

        data_torch = torch.from_numpy(data)
        if self.metric == "cosine":
            data_norm = torch.linalg.norm(data_torch, dim=1, keepdim=True)
            data_torch = data_torch / (data_norm + self.epsilon) # Use self.epsilon
            # data_norms_torch is not used for cosine
            data_norms_torch = torch.empty(data_torch.shape[0], device=data_torch.device, dtype=data_torch.dtype) 
        elif self.metric == "euclidean":
            data_norms_torch = (data_torch**2).sum(dim=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


        # We'll do a chunked assignment pass, similar to the main loop, but no centroid updates
        centroids_torch = torch.from_numpy(self.centroids)
        if self.metric == "cosine":
            centroid_norm = torch.linalg.norm(centroids_torch, dim=1, keepdim=True)
            centroids_torch = centroids_torch / (centroid_norm + self.epsilon) # Use self.epsilon
        
        centroids_torch = centroids_torch.to(device=self.device, dtype=torch.float32)
        
        if self.metric == "euclidean":
            centroid_norms = (centroids_torch**2).sum(dim=1)
        # For cosine, centroid_norms is not used in the same way. 
        # The triton kernel expects it, but triton is disabled for cosine.
        # For the non-triton path, it's not used. So we can leave it uninitialized or zeros for cosine.
        elif self.metric == "cosine" and self.use_triton:
             # This case should not happen based on _kmeans_torch_double_chunked logic
             # but as a safeguard, initialize to zeros if triton were ever enabled for cosine
            centroid_norms = torch.zeros(centroids_torch.shape[0], device=self.device, dtype=torch.float32)


        n_samples = data_torch.shape[0]
        labels = torch.empty(n_samples, dtype=torch.long, device="cpu")

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + self.chunk_size_data, n_samples)

            data_chunk = data_torch[start_idx:end_idx].to(device=self.device, dtype=torch.float32, non_blocking=True)
            if self.metric == "euclidean":
                data_chunk_norms = data_norms_torch[start_idx:end_idx].to(
                    device=self.device, dtype=torch.float32, non_blocking=True
                )
            batch_size = data_chunk.size(0)
            best_ids = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

            if self.use_triton and self.metric == "euclidean": # Triton only for euclidean
                triton_kmeans(
                    data_chunk,
                    data_chunk_norms, # This is correct for euclidean
                    centroids_torch,
                    centroid_norms, # This is correct for euclidean
                    best_ids,
                )
            else:
                best_dist = torch.full((batch_size,), float("inf"), device=self.device, dtype=torch.float32)
                c_start = 0
                k = centroids_torch.shape[0]
                while c_start < k:
                    c_end = min(c_start + self.chunk_size_centroids, k)
                    centroid_chunk = centroids_torch[c_start:c_end]
                    
                    if self.metric == "euclidean":
                        centroid_chunk_norms = centroid_norms[c_start:c_end]
                        dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                        dist_chunk = dist_chunk.addmm_(data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0)
                    elif self.metric == "cosine":
                        # For cosine, best is highest dot product, so we minimize 1 - dot_product
                        dist_chunk = 1 - torch.matmul(data_chunk, centroid_chunk.t())


                    local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                    improved_mask = local_min_vals < best_dist
                    best_dist[improved_mask] = local_min_vals[improved_mask]
                    best_ids[improved_mask] = c_start + local_min_ids[improved_mask]
                    c_start = c_end

            labels[start_idx:end_idx] = best_ids.to("cpu")
            start_idx = end_idx

        return labels.numpy()

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Chains fit and predict, once again inspired by the great scikit-learn.
        """
        self.fit(data)
        return self.predict(data)
