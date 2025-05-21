from __future__ import annotations

from contextlib import contextmanager

import torch
import triton
import triton.language as tl


@contextmanager
def device_guard(tensor: torch.Tensor):
    """Context manager to ensure that the Triton kernel launches on the correct device."""
    if tensor.device.type == "cuda":  # NVIDIA or AMD/ROCm
        with torch.cuda.device_of(tensor):
            yield
    elif tensor.device.type == "xpu":  # Intel GPUs
        with torch.xpu.device_of(tensor):
            yield
    else:  # CPU or other back-ends
        yield


@triton.heuristics(
    {
        "BLOCK_M": lambda x: 128 if x["D"] <= 384 else 64,
        "BLOCK_N": lambda x: 128 if x["D"] <= 384 else 64,
        "BLOCK_K": lambda x: 16 if x["D"] <= 32 or x["D"] > 384 else 32,
        "GROUP_SIZE_M": lambda x: 8 if x["D"] <= 32 else 16,
        "num_warps": lambda x: 4,
    }
)
@triton.jit
def _kmeans_kernel(
    x_ptr,
    x_norm_ptr,
    c_ptr,
    c_norm_ptr,
    best_dist_ptr,
    best_idx_ptr,
    B,
    C,
    D: tl.constexpr,
    METRIC: tl.constexpr, # "euclidean" or "cosine"
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPSILON: tl.constexpr = 1e-8, # Added epsilon
):
    """
    Triton kernel for computing distances and assigning points to nearest centroids.

    Parameters
    ----------
    x_ptr
        Pointer to the data chunk tensor (n_samples_chunk, n_features).
    x_norm_ptr
        Pointer to the tensor of squared L2 norms for data points in `x_ptr`.
        Used only if `METRIC` is "EUCLIDEAN".
    c_ptr
        Pointer to the centroids tensor (k, n_features).
    c_norm_ptr
        Pointer to the tensor of squared L2 norms for centroids in `c_ptr`.
        Used only if `METRIC` is "EUCLIDEAN".
    best_dist_ptr
        Pointer to a tensor for storing the minimum distance found so far for each data point.
    best_idx_ptr
        Pointer to a tensor for storing the index of the closest centroid for each data point.
    B : int
        Number of data points in the current chunk (`data_chunk.shape[0]`).
    C : int
        Number of centroids (`centroids.shape[0]`).
    D : tl.constexpr
        Dimensionality of the data and centroids.
    METRIC : tl.constexpr
        Compile-time constant string indicating the metric: "EUCLIDEAN" or "COSINE".
        - If "EUCLIDEAN", computes squared Euclidean distance: `x_norm - 2*dot(x,c) + c_norm`.
          `x_norm_ptr` and `c_norm_ptr` are used.
        - If "COSINE", computes `1 - dot(x,c) / (||x||*||c||)`. Vectors `x` and `c` (rows from
          `x_ptr` and `c_ptr`) are normalized internally using their full L2 norms calculated on-the-fly
          within this kernel. `x_norm_ptr` and `c_norm_ptr` are ignored.
    BLOCK_M : tl.constexpr
        Tile size for the data dimension.
    BLOCK_N : tl.constexpr
        Tile size for the centroid dimension.
    BLOCK_K : tl.constexpr
        Tile size for the feature dimension (inner dimension of dot product).
    GROUP_SIZE_M : tl.constexpr
        Group size for parallel reduction over data dimension.
    EPSILON : tl.constexpr, default=1e-8
        Small constant to add to denominators during L2 normalization (when `METRIC`="COSINE")
        to prevent division by zero or issues with zero-norm vectors.
    """
    # Map flat CTA id to (pid_m, pid_n) in “grouped” launch order
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(B, BLOCK_M)  # row-tiles
    num_pid_n = tl.cdiv(C, BLOCK_N)  # centroid-tiles

    # Super-group into GROUP_SIZE_M blocks to minimize loading from global memory
    num_pid_in_grp = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_grp) * GROUP_SIZE_M
    group_rows = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_grp) % group_rows)  # row-tile index
    pid_n = (pid % num_pid_in_grp) // group_rows  # centroid-tile index

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    row_mask = rows < B
    col_mask = cols < C

    # pipelined K‑loop
    dot_acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    
    if METRIC == "cosine":
        sum_sq_x = tl.zeros([BLOCK_M, 1], tl.float32)
        sum_sq_c = tl.zeros([BLOCK_N, 1], tl.float32) # Sum of squares for each centroid vector

    # compute matmul tiled across SMs
    for k0 in range(0, D, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # load X slice
        x_ptrs = x_ptr + rows[:, None] * D + k_range[None, :]
        # xk shape: [BLOCK_M, BLOCK_K]
        xk = tl.load(x_ptrs, mask=row_mask[:, None] & (k_range[None, :] < D)).to(tl.float16)

        # load C slice
        c_ptrs = c_ptr + cols[:, None] * D + k_range[None, :]
        # ck shape: [BLOCK_N, BLOCK_K]
        ck = tl.load(c_ptrs, mask=col_mask[:, None] & (k_range[None, :] < D)).to(tl.float16)
        
        dot_acc += tl.dot(xk, tl.trans(ck), out_dtype=tl.float32)

        if METRIC == "cosine":
            # Accumulate sum of squares for normalization
            # For xk (data points): sum over K dimension for each of M rows
            sum_sq_x += tl.sum(xk * xk, axis=1, keepdims=True).to(tl.float32)
            # For ck (centroids): sum over K dimension for each of N rows
            sum_sq_c += tl.sum(ck * ck, axis=1, keepdims=True).to(tl.float32)


    if METRIC == "euclidean":
        # load precomputed norms for Euclidean distance
        x_n = tl.load(x_norm_ptr + rows, mask=row_mask, other=0.0)  # [BM]
        c_n = tl.load(c_norm_ptr + cols, mask=col_mask, other=0.0)  # [BN]
        # finish distance formula: d^2 = x_n - 2*dot_acc + c_n
        dist = tl.fma(dot_acc, -2.0, x_n[:, None] + c_n[None, :])  # [BM, BN]
    elif METRIC == "cosine":
        norm_x = tl.sqrt(sum_sq_x) + EPSILON # [BLOCK_M, 1]
        norm_c = tl.sqrt(sum_sq_c) + EPSILON # [BLOCK_N, 1]
        
        # dot_acc is [BLOCK_M, BLOCK_N]
        # norm_x is [BLOCK_M, 1], norm_c is [BLOCK_N, 1]
        # We need denominator of shape [BLOCK_M, BLOCK_N]
        # tl.trans(norm_c) is [1, BLOCK_N]
        denominator = norm_x * tl.trans(norm_c) # [BLOCK_M, BLOCK_N]
        
        normalized_dot_acc = dot_acc / denominator
        dist = 1.0 - normalized_dot_acc # [BM, BN]
    # else: # Should be caught by Python checks or METRIC.upper() would fail compile

    # local arg‑min (inside this tile)
    # For cosine, we minimize 1 - sim, which is equivalent to maximizing similarity.
    tile_min, tile_idx = tl.min(dist, axis=1, return_indices=True)

    # compete with global best using atomics
    prev = tl.atomic_min(best_dist_ptr + rows, tile_min, mask=row_mask)
    improved = tile_min < prev

    # update best_ids
    tl.store(best_idx_ptr + rows, tl.where(improved, col_start + tile_idx, tl.load(best_idx_ptr + rows)), mask=row_mask)


def triton_kmeans(
    data_chunk: torch.Tensor,
    data_chunk_norms: torch.Tensor,
    centroids: torch.Tensor,
    centroids_sqnorm: torch.Tensor,
    best_ids: torch.Tensor,
    metric: str = "euclidean",
):
    """
    Performs a single pass of assigning data points to the nearest centroids using a Triton kernel.
    This function is a wrapper around `_kmeans_kernel`.

    Parameters
    ----------
    data_chunk : torch.Tensor
        A chunk of input data, shape (n_samples_chunk, n_features).
    data_chunk_norms : torch.Tensor
        Squared L2 norms of the `data_chunk` points. Shape (n_samples_chunk,).
        Used by the kernel only if `metric` is "euclidean". If `metric` is "cosine",
        this parameter is passed to the kernel but effectively ignored as the kernel normalizes internally.
    centroids : torch.Tensor
        Current centroids, shape (k, n_features).
    centroids_sqnorm : torch.Tensor
        Squared L2 norms of the `centroids`. Shape (k,).
        Used by the kernel only if `metric` is "euclidean". If `metric` is "cosine",
        this parameter is passed to the kernel but effectively ignored for the same reason.
    best_ids : torch.Tensor
        Output tensor to store the indices of the best (closest) centroid for each
        point in `data_chunk`. Shape (n_samples_chunk,).
    metric : str, default="euclidean"
        The distance metric to use. Can be "euclidean" or "cosine".
        This string is converted to uppercase and passed to the Triton kernel
        as a compile-time constant (`METRIC`).
        - "euclidean": Standard Euclidean distance. `data_chunk_norms` and `centroids_sqnorm` are used by the kernel.
        - "cosine": Cosine similarity based distance. `data_chunk_norms` and `centroids_sqnorm` are
          passed to the kernel but effectively ignored, as vector normalization and the
          cosine distance calculation are handled entirely within the kernel using the raw `data_chunk`
          and `centroids`.
    """
    B, D = data_chunk.shape
    C = centroids.shape[0]

    # Metric validation happens here before passing to JITed kernel
    if metric.upper() not in ["EUCLIDEAN", "COSINE"]:
        raise ValueError(f"Unsupported metric for Triton kernel: {metric}. Must be 'euclidean' or 'cosine'.")
    
    best_dist = torch.full((B,), 1e38, device=data_chunk.device, dtype=torch.float32)

    def grid(meta):
        # Consider D in heuristic for BLOCK_K if not already done
        # Heuristics are defined above the kernel
        return (triton.cdiv(B, meta["BLOCK_M"]) * triton.cdiv(C, meta["BLOCK_N"]),)  # 1D grid

    # Without this Triton always tries to launch from device:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with device_guard(data_chunk):
        _kmeans_kernel[grid](
            data_chunk,
            data_chunk_norms,
            centroids,
            centroids_sqnorm, # Effectively ignored by kernel if metric is cosine
            best_dist,
            best_ids,
            B,
            C,
            D,
            METRIC=metric.upper(), 
            # EPSILON can be passed here if we don't want it as a kernel default, 
            # but as a default tl.constexpr it's fine.
        )
