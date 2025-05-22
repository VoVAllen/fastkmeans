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


def _kmeans_plusplus_init(data: torch.Tensor, k: int, device: torch.device, seed: int) -> torch.Tensor:
    """
    KMeans++ initialization.

    Parameters
    ----------
    data : torch.Tensor
        Input data of shape (n_samples, n_features) on the target device.
    k : int
        Number of centroids to select.
    device : torch.device
        The device where the centroids should be.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        The k centroids, shape (k, n_features), on the specified device and with the same dtype as data.
    """
    torch.manual_seed(seed)
    n_samples, n_features = data.shape
    centroids = torch.empty((k, n_features), dtype=data.dtype, device=device)

    # 1. Select the first centroid randomly
    first_centroid_idx = torch.randint(n_samples, (1,)).item()
    centroids[0] = data[first_centroid_idx]

    if k == 1:
        return centroids

    # 2. For each of the remaining k-1 centroids
    for i in range(1, k):
        # a. For each data point, calculate its squared Euclidean distance to the *nearest* already selected centroid
        # Equivalent to: dists_sq = torch.cdist(data, centroids[:i]).min(dim=1).values ** 2
        # but cdist can be slow for large data, manual computation is often faster and uses less memory
        
        # Expand centroids[:i] and data for broadcasting
        # data shape: (n_samples, n_features)
        # centroids_so_far shape: (i, n_features)
        # We want to compute distances from each point in data to each centroid in centroids_so_far
        
        # (n_samples, 1, n_features) - (1, i, n_features) -> (n_samples, i, n_features)
        diffs = data.unsqueeze(1) - centroids[:i].unsqueeze(0)
        dists_sq_all_centroids = (diffs ** 2).sum(dim=2) # (n_samples, i)
        
        min_dists_sq, _ = torch.min(dists_sq_all_centroids, dim=1) # (n_samples,)

        # b. Select the next centroid from the data points with probability proportional to D_sq(x)
        if torch.all(min_dists_sq == 0):
            # This can happen if k is larger than the number of unique points
            # or if points are chosen that are identical to existing centroids.
            # In this case, pick remaining centroids randomly to avoid errors with multinomial.
            # It might be better to pick from points that are not yet centroids,
            # but random selection is simpler and robust.
            num_remaining_centroids = k - i
            random_indices = torch.randperm(n_samples, device=device)[:num_remaining_centroids]
            centroids[i:] = data[random_indices]
            break # Exit the loop as all remaining centroids are filled

        # Ensure probabilities are not zero for all points, can happen if some points are identical.
        # Add a small epsilon if all distances are zero to avoid issues with multinomial if all dists are 0.
        # However, the `if torch.all(min_dists_sq == 0)` check above should handle this.
        # If min_dists_sq sums to 0, it means all points are identical to chosen centroids,
        # this case is handled. If not all are zero, but some are, multinomial handles it.

        probabilities = min_dists_sq / torch.sum(min_dists_sq)
        
        # Check for NaN or Inf in probabilities which can occur if min_dists_sq contains NaNs or Infs,
        # or if sum is zero.
        if torch.isnan(probabilities).any() or torch.isinf(probabilities).any() or torch.sum(min_dists_sq) == 0:
            # Fallback to random sampling if probabilities are problematic
            # This could happen if all points are identical to centroids already selected.
            num_remaining_centroids = k - i
            random_indices = torch.randperm(n_samples, device=device)[:num_remaining_centroids]
            centroids[i:] = data[random_indices]
            break

        next_centroid_idx = torch.multinomial(probabilities, 1).item()
        centroids[i] = data[next_centroid_idx]

    return centroids


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
    init_method: str = "random",
    seed: int = 0,
):
    """
    An efficient kmeans implementation that minimises OOM risks on modern hardware by using conversative double chunking.

    Parameters
    ----------
    data : torch.Tensor
        Input data tensor.
    data_norms : torch.Tensor
        Squared L2 norms of the input data.
    k : int
        Number of clusters.
    device : torch.device
        Target device for computation.
    dtype : torch.dtype | None, default=None
        Target dtype for computation.
    max_iters : int, default=25
        Maximum number of iterations.
    tol : float, default=1e-8
        Tolerance for convergence.
    chunk_size_data : int, default=50_000
        Chunk size for data processing.
    chunk_size_centroids : int, default=10_000
        Chunk size for centroid processing.
    max_points_per_centroid : int, default=256
        Maximum points per centroid for subsampling.
    verbose : bool, default=False
        Enable verbose logging.
    use_triton : bool | None, default=None
        Enable Triton kernels.
    init_method : str, default='random'
        Centroid initialization method ('random' or 'kmeans++').
    seed : int, default=0
        Random seed for initialization.

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
        data_norms = data_norms[indices]
        n_samples = target_n_samples
        del perm, indices

    if n_samples < k:
        raise ValueError(f"Number of training points ({n_samples}) is less than k ({k}).")

    # Centroid initialization
    if init_method == "kmeans++":
        # Data for _kmeans_plusplus_init should be on the target device
        data_for_init = data.to(device=device, dtype=dtype if dtype is not None else data.dtype)
        centroids = _kmeans_plusplus_init(data_for_init, k, device, seed)
        # Ensure centroids are correctly typed and on device, _kmeans_plusplus_init should handle this, but being explicit.
        centroids = centroids.to(device=device, dtype=dtype)
    elif init_method == "random":
        # Ensure randperm happens on the device of the data if data is still on CPU
        # or on the target device if data has already been moved.
        # Since `data` at this point is the (potentially subsampled) data, which might be on CPU or GPU,
        # and `device` is the target computation device.
        # For simplicity and consistency, let's ensure data is on the target device before randperm if it's used for indexing.
        data_on_target_device = data.to(device=device)
        rand_indices = torch.randperm(n_samples, device=device)[:k]
        centroids = data_on_target_device[rand_indices].clone().to(device=device, dtype=dtype)
    else:
        # Fallback or error for unknown init_method, though FastKMeans class should prevent this.
        # For now, let's default to random for safety if an unexpected value gets here.
        # This case should ideally not be reached if FastKMeans validates `init`.
        torch.manual_seed(seed) # Ensure seed is respected for random init
        data_on_target_device = data.to(device=device)
        rand_indices = torch.randperm(n_samples, device=device)[:k]
        centroids = data_on_target_device[rand_indices].clone().to(device=device, dtype=dtype)

    prev_centroids = centroids.clone()

    labels = torch.empty(n_samples, dtype=torch.int64, device="cpu")  # Keep labels on CPU

    for iteration in range(max_iters):
        iteration_start_time = time.time()

        centroid_norms = (centroids**2).sum(dim=1)
        cluster_sums = torch.zeros((k, n_features), device=device, dtype=torch.float32)
        cluster_counts = torch.zeros((k,), device=device, dtype=torch.float32)

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + chunk_size_data, n_samples)

            data_chunk = data[start_idx:end_idx].to(device=device, dtype=dtype, non_blocking=True)
            data_chunk_norms = data_norms[start_idx:end_idx].to(device=device, dtype=dtype, non_blocking=True)
            batch_size = data_chunk.size(0)
            best_ids = torch.zeros((batch_size,), device=device, dtype=torch.int64)

            if use_triton:
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
                    centroid_chunk_norms = centroid_norms[c_start:c_end]

                    dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                    dist_chunk = dist_chunk.addmm_(data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0)

                    local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                    improved_mask = local_min_vals < best_dist
                    best_dist[improved_mask] = local_min_vals[improved_mask]
                    best_ids[improved_mask] = c_start + local_min_ids[improved_mask]

                    c_start = c_end

            cluster_sums.index_add_(0, best_ids, data_chunk.float())
            cluster_counts.index_add_(0, best_ids, torch.ones_like(best_ids, device=device, dtype=torch.float32))

            labels[start_idx:end_idx] = best_ids.to("cpu", non_blocking=True)
            start_idx = end_idx

        new_centroids = torch.zeros_like(centroids, device=device, dtype=dtype)
        non_empty = cluster_counts > 0
        new_centroids[non_empty] = (cluster_sums[non_empty] / cluster_counts[non_empty].unsqueeze(1)).to(dtype=dtype)

        empty_ids = (~non_empty).nonzero(as_tuple=True)[0]
        if len(empty_ids) > 0:
            reinit_indices = torch.randint(0, n_samples, (len(empty_ids),), device="cpu")
            random_data = data[reinit_indices].to(device=device, dtype=dtype, non_blocking=True)
            new_centroids[empty_ids] = random_data

        shift = torch.norm(new_centroids - prev_centroids.to(new_centroids.device), dim=1).sum().item()
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
    niter : int, default=20
        Maximum number of iterations.
    tol : float, default=1e-4
        Stopping threshold for centroid movement.
    gpu : bool, default=True
        Whether to force GPU usage if available. If False, CPU is used.
    seed : int, default=0
        Random seed for centroid initialization and (if needed) subsampling.
    max_points_per_centroid : int, optional, default=1_000_000_000
        If n_samples > k * max_points_per_centroid, the data will be subsampled to exactly
        k * max_points_per_centroid points before clustering.
    chunk_size_data : int, default=10,2400
        Chunk size along the data dimension for assignment/update steps.
    chunk_size_centroids : int, default=10,240
        Chunk size along the centroid dimension for assignment/update steps.
    use_triton : bool | None, default=None
       Use the fast Triton backend for the assignment/update steps.
       If None, the Triton backend will be enabled for modern GPUs.
    init : str, default='random'
        Method for centroid initialization. Valid options are 'random' and 'kmeans++'.
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
        init: str = "random",
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
        if init not in ["random", "kmeans++"]:
            raise ValueError(f"Invalid init method: {init}. Valid options are 'random' and 'kmeans++'.")
        self.init = init

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
        data_norms_torch = (data_torch**2).sum(dim=1)

        device = _get_device(self.device)
        if device == "cuda" and self.pin_gpu_memory:
            data_torch = data_torch.pin_memory()
            data_norms_torch = data_norms_torch.pin_memory()

        centroids, _ = _kmeans_torch_double_chunked(
            data_torch,
            data_norms_torch,
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
            init_method=self.init,
            seed=self.seed,
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
        data_norms_torch = (data_torch**2).sum(dim=1)

        # We'll do a chunked assignment pass, similar to the main loop, but no centroid updates
        centroids_torch = torch.from_numpy(self.centroids)
        centroids_torch = centroids_torch.to(device=self.device, dtype=torch.float32)
        centroid_norms = (centroids_torch**2).sum(dim=1)

        n_samples = data_torch.shape[0]
        labels = torch.empty(n_samples, dtype=torch.long, device="cpu")

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + self.chunk_size_data, n_samples)

            data_chunk = data_torch[start_idx:end_idx].to(device=self.device, dtype=torch.float32, non_blocking=True)
            data_chunk_norms = data_norms_torch[start_idx:end_idx].to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            batch_size = data_chunk.size(0)
            best_ids = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

            if self.use_triton:
                triton_kmeans(
                    data_chunk,
                    data_chunk_norms,
                    centroids_torch,
                    centroid_norms,
                    best_ids,
                )
            else:
                best_dist = torch.full((batch_size,), float("inf"), device=self.device, dtype=torch.float32)
                c_start = 0
                k = centroids_torch.shape[0]
                while c_start < k:
                    c_end = min(c_start + self.chunk_size_centroids, k)
                    centroid_chunk = centroids_torch[c_start:c_end]
                    centroid_chunk_norms = centroid_norms[c_start:c_end]

                    dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                    dist_chunk = dist_chunk.addmm_(data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0)

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
