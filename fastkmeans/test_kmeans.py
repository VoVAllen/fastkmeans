import pytest
import torch
import numpy as np
from sklearn.preprocessing import normalize as sk_normalize

from fastkmeans.kmeans import FastKMeans, _get_device, HAS_TRITON

# Helper to check if a CUDA device is available
CUDA_AVAILABLE = torch.cuda.is_available()

def generate_spherical_data(n_samples, n_features, n_clusters, seed=42):
    """
    Generates data points on the surface of a unit sphere,
    potentially clustered. For simplicity, we'll generate points
    and then normalize them. For distinct clusters, we can generate
    from different means before normalization.
    """
    np.random.seed(seed)
    if n_clusters == 1:
        # Random points, will be normalized
        data = np.random.rand(n_samples, n_features) - 0.5
    else:
        data = []
        samples_per_cluster = n_samples // n_clusters
        for i in range(n_clusters):
            # Generate points around a random mean, then normalize globally
            mean = np.random.rand(n_features) * 10 - 5 
            cluster_data = np.random.randn(samples_per_cluster, n_features) + mean
            data.append(cluster_data)
        
        remaining_samples = n_samples - (samples_per_cluster * n_clusters)
        if remaining_samples > 0:
            mean = np.random.rand(n_features) * 10 - 5
            cluster_data = np.random.randn(remaining_samples, n_features) + mean
            data.append(cluster_data)
        
        data = np.vstack(data)

    data_normalized = sk_normalize(data, norm='l2', axis=1)
    return data_normalized.astype(np.float32)


@pytest.fixture
def default_device_str():
    return "cuda" if CUDA_AVAILABLE else "cpu"

# Basic test structure
# We will test different combinations of parameters using parametrize
# For use_triton=True, it implies GPU. We'll skip if no GPU.
# For use_triton=False, we can test on CPU and GPU.

# Define parameter sets
param_configs = []
# PyTorch CPU
param_configs.append(pytest.param({"metric": "cosine", "use_triton": False, "device_str": "cpu", "dtype": torch.float32}, id="cosine-torch-cpu-fp32"))
if CUDA_AVAILABLE:
    # PyTorch GPU
    param_configs.append(pytest.param({"metric": "cosine", "use_triton": False, "device_str": "cuda", "dtype": torch.float32}, id="cosine-torch-gpu-fp32"))
    param_configs.append(pytest.param({"metric": "cosine", "use_triton": False, "device_str": "cuda", "dtype": torch.float16}, id="cosine-torch-gpu-fp16", marks=pytest.mark.skipif(not _get_device("cuda").type == "cuda" or not torch.cuda.is_bf16_supported(), reason="FP16/BF16 on CUDA required"))) # Assuming float16 for testing, bf16 might be better
    # Triton GPU
    if HAS_TRITON:
        param_configs.append(pytest.param({"metric": "cosine", "use_triton": True, "device_str": "cuda", "dtype": torch.float32}, id="cosine-triton-gpu-fp32"))
        param_configs.append(pytest.param({"metric": "cosine", "use_triton": True, "device_str": "cuda", "dtype": torch.float16}, id="cosine-triton-gpu-fp16", marks=pytest.mark.skipif(not _get_device("cuda").type == "cuda" or not torch.cuda.is_bf16_supported(), reason="FP16/BF16 on CUDA required for Triton test")))


@pytest.mark.parametrize("config", param_configs)
def test_spherical_kmeans_init_and_train(config, default_device_str):
    if config["use_triton"] and not CUDA_AVAILABLE:
        pytest.skip("Triton test requires CUDA.")
    if config["device_str"] == "cuda" and not CUDA_AVAILABLE:
        pytest.skip("CUDA test requires CUDA.")
    if config["dtype"] == torch.float16 and config["device_str"] == "cpu":
        pytest.skip("FP16 is not well supported on CPU for this test.")
        
    device = _get_device(config["device_str"])
    if config["dtype"] == torch.float16 and device.type == "cuda" and not torch.cuda.is_bf16_supported(): # Checking for bf16 as a proxy for general float16 support on device
         pytest.skip(f"Device {device} does not support float16/bfloat16 sufficiently for this test.")


    n_samples, n_features, k = 100, 10, 3
    data = generate_spherical_data(n_samples, n_features, k)

    model = FastKMeans(
        d=n_features,
        k=k,
        metric=config["metric"],
        use_triton=config["use_triton"],
        device=config["device_str"], # Correctly pass device string
        dtype=config["dtype"],
        niter=5 # Keep iterations low for speed
    )
    
    model.train(data)

    assert model.centroids is not None
    assert model.centroids.shape == (k, n_features)
    
    # Check if centroids are normalized
    centroid_norms = np.linalg.norm(model.centroids, axis=1)
    assert np.allclose(centroid_norms, 1.0, atol=1e-5 if config["dtype"] == torch.float32 else 1e-2), \
        f"Centroids not normalized. Norms: {centroid_norms}"


@pytest.mark.parametrize("config", param_configs)
def test_spherical_kmeans_k_one(config, default_device_str):
    if config["use_triton"] and not CUDA_AVAILABLE:
        pytest.skip("Triton test requires CUDA.")
    if config["device_str"] == "cuda" and not CUDA_AVAILABLE:
        pytest.skip("CUDA test requires CUDA.")
    if config["dtype"] == torch.float16 and config["device_str"] == "cpu":
        pytest.skip("FP16 is not well supported on CPU for this test.")

    device = _get_device(config["device_str"])
    if config["dtype"] == torch.float16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
         pytest.skip(f"Device {device} does not support float16/bfloat16 sufficiently for this test.")

    n_samples, n_features, k = 100, 10, 1
    data = generate_spherical_data(n_samples, n_features, k)

    model = FastKMeans(
        d=n_features,
        k=k,
        metric=config["metric"],
        use_triton=config["use_triton"],
        device=config["device_str"],
        dtype=config["dtype"],
        niter=5
    )
    model.train(data)

    assert model.centroids.shape == (k, n_features)
    centroid_norms = np.linalg.norm(model.centroids, axis=1)
    assert np.allclose(centroid_norms, 1.0, atol=1e-5 if config["dtype"] == torch.float32 else 1e-2)

    # For k=1, the centroid should be the normalized mean of the (already normalized) data.
    # The internal algorithm takes random data points as init, then averages.
    # If data is already normalized, its average, when re-normalized, is the centroid.
    
    # For cosine similarity, the "mean" is more complex.
    # However, all points should be assigned to cluster 0.
    labels = model.predict(data)
    assert np.all(labels == 0)


@pytest.mark.parametrize("config", param_configs)
def test_spherical_kmeans_identical_points(config, default_device_str):
    if config["use_triton"] and not CUDA_AVAILABLE:
        pytest.skip("Triton test requires CUDA.")
    if config["device_str"] == "cuda" and not CUDA_AVAILABLE:
        pytest.skip("CUDA test requires CUDA.")
    if config["dtype"] == torch.float16 and config["device_str"] == "cpu":
        pytest.skip("FP16 is not well supported on CPU for this test.")
        
    device = _get_device(config["device_str"])
    if config["dtype"] == torch.float16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
         pytest.skip(f"Device {device} does not support float16/bfloat16 sufficiently for this test.")

    n_samples, n_features, k = 100, 10, 3
    # Create identical, normalized data
    point = sk_normalize(np.random.rand(1, n_features), norm='l2', axis=1).astype(np.float32)
    data = np.repeat(point, n_samples, axis=0)

    model = FastKMeans(
        d=n_features,
        k=k,
        metric=config["metric"],
        use_triton=config["use_triton"],
        device=config["device_str"],
        dtype=config["dtype"],
        niter=10 # More iters to ensure convergence
    )
    model.train(data)

    assert model.centroids.shape == (k, n_features)
    centroid_norms = np.linalg.norm(model.centroids, axis=1)
    assert np.allclose(centroid_norms, 1.0, atol=1e-5 if config["dtype"] == torch.float32 else 1e-2)

    # All centroids should be very close to the original point
    # Due to random init from the data points themselves, all centroids will be that point.
    for i in range(k):
        assert np.allclose(model.centroids[i], point[0], atol=1e-4 if config["dtype"] == torch.float32 else 1e-2), \
            f"Centroid {i} is not close to the data point. Centroid: {model.centroids[i]}, Point: {point[0]}"

    labels = model.predict(data)
    # All points should be clustered to one of the (identical) centroids.
    # It's possible they get split if centroids are numerically unstable, but they should all be ~the same.
    # Forcing k=1 here would be a stronger test of this specific aspect, but k=3 is fine.
    # We check that all predicted labels correspond to centroids that are very close to the original point.
    for i in range(n_samples):
        assigned_centroid = model.centroids[labels[i]]
        assert np.allclose(assigned_centroid, point[0], atol=1e-4 if config["dtype"] == torch.float32 else 1e-2)


@pytest.mark.parametrize("config", param_configs)
def test_spherical_kmeans_predict_method(config, default_device_str):
    if config["use_triton"] and not CUDA_AVAILABLE:
        pytest.skip("Triton test requires CUDA.")
    if config["device_str"] == "cuda" and not CUDA_AVAILABLE:
        pytest.skip("CUDA test requires CUDA.")
    if config["dtype"] == torch.float16 and config["device_str"] == "cpu":
        pytest.skip("FP16 is not well supported on CPU for this test.")

    device = _get_device(config["device_str"])
    if config["dtype"] == torch.float16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
         pytest.skip(f"Device {device} does not support float16/bfloat16 sufficiently for this test.")

    n_samples, n_features, k = 100, 10, 3
    train_data = generate_spherical_data(n_samples, n_features, k, seed=42)
    predict_data = generate_spherical_data(n_samples // 2, n_features, k, seed=43) # Different data

    model = FastKMeans(
        d=n_features,
        k=k,
        metric=config["metric"],
        use_triton=config["use_triton"],
        device=config["device_str"],
        dtype=config["dtype"],
        niter=5
    )
    model.train(train_data)
    
    assert model.centroids is not None # Ensure training happened

    labels = model.predict(predict_data)
    assert labels.shape == (n_samples // 2,)
    assert labels.min() >= 0
    assert labels.max() < k

    # Check that predict itself doesn't alter centroids (it shouldn't for cosine)
    original_centroids = model.centroids.copy()
    _ = model.predict(predict_data) # predict again
    assert np.allclose(model.centroids, original_centroids)


# More advanced test: separable clusters
# This is harder to guarantee perfect separation due to random init
# but we can check if the majority of points are assigned correctly.
def generate_specific_spherical_clusters(n_samples_per_cluster, n_features, seed=42):
    np.random.seed(seed)
    # Define some base vectors for clusters
    base_vectors = np.eye(n_features)[:3] # Max 3 clusters for simplicity if n_features >=3
    if n_features < 3: # Fallback for low dim
        base_vectors = np.random.randn(3, n_features)
        base_vectors = sk_normalize(base_vectors, norm='l2', axis=1)

    n_clusters = base_vectors.shape[0]
    
    all_data = []
    true_labels = []
    
    for i in range(n_clusters):
        # Generate points "around" the base vector by adding small noise and re-normalizing
        # Adding noise in tangent space is more rigorous but this is simpler for a test.
        noise = np.random.randn(n_samples_per_cluster, n_features) * 0.1 # Small noise
        cluster_points = base_vectors[i] + noise
        cluster_points = sk_normalize(cluster_points, norm='l2', axis=1)
        all_data.append(cluster_points)
        true_labels.extend([i] * n_samples_per_cluster)
        
    data = np.vstack(all_data).astype(np.float32)
    return data, np.array(true_labels), base_vectors[:n_clusters]


@pytest.mark.parametrize("config", param_configs)
def test_spherical_kmeans_separable(config, default_device_str):
    if config["use_triton"] and not CUDA_AVAILABLE:
        pytest.skip("Triton test requires CUDA.")
    if config["device_str"] == "cuda" and not CUDA_AVAILABLE:
        pytest.skip("CUDA test requires CUDA.")
    if config["dtype"] == torch.float16 and config["device_str"] == "cpu":
        pytest.skip("FP16 is not well supported on CPU for this test.")
        
    device = _get_device(config["device_str"])
    if config["dtype"] == torch.float16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
         pytest.skip(f"Device {device} does not support float16/bfloat16 sufficiently for this test.")

    n_samples_per_cluster, n_features = 50, 3 # Using 3 features for easy distinct clusters
    k = 3 
    
    data, true_labels, true_centers_approx = generate_specific_spherical_clusters(n_samples_per_cluster, n_features)
    
    model = FastKMeans(
        d=n_features,
        k=k,
        metric=config["metric"],
        use_triton=config["use_triton"],
        device=config["device_str"],
        dtype=config["dtype"],
        niter=15, # More iterations for separable clusters
        seed=42 # Set seed for reproducibility of init
    )
    model.train(data)

    assert model.centroids is not None
    centroid_norms = np.linalg.norm(model.centroids, axis=1)
    assert np.allclose(centroid_norms, 1.0, atol=1e-5 if config["dtype"] == torch.float32 else 1e-2)

    # Check if learned centroids are close to the true generating centers (after matching)
    # This is tricky due to label permutation.
    # A simpler check: for each true cluster, find the majority predicted label.
    # Then check if the centroid for that predicted label is close to the true_center.
    
    predicted_labels = model.predict(data)
    
    # Check purity or Adjusted Rand Index if we want to be more rigorous
    # For now, let's check if a high percentage of points are clustered "reasonably"
    # i.e. points from the same original cluster mostly go to the same predicted cluster.
    
    # A qualitative check: cosine similarity of learned centroids to true_centers_approx
    # For each true center, find the learned centroid with highest cosine similarity
    matched_learned_centroids = []
    remaining_learned_indices = list(range(k))
    
    for i in range(k):
        true_center = true_centers_approx[i]
        similarities = [np.dot(true_center, model.centroids[j]) for j in remaining_learned_indices]
        best_match_idx_in_remaining = np.argmax(similarities)
        best_match_original_idx = remaining_learned_indices.pop(best_match_idx_in_remaining)
        matched_learned_centroids.append(model.centroids[best_match_original_idx])
        
        # Expect high similarity for matched centroids
        # The threshold here is heuristic.
        assert np.dot(true_center, model.centroids[best_match_original_idx]) > 0.7, \
            f"True center {i} not well matched. Similarity: {np.dot(true_center, model.centroids[best_match_original_idx])}"

    # Check if inertia (sum of 1 - cosine_similarity to closest centroid) is low
    # This is essentially what the algorithm minimizes
    total_cosine_similarity = 0
    for i in range(data.shape[0]):
        point = data[i]
        assigned_centroid = model.centroids[predicted_labels[i]]
        total_cosine_similarity += np.dot(point, assigned_centroid)
    
    mean_cosine_similarity = total_cosine_similarity / data.shape[0]
    # For well-separated spherical data, expect high average similarity
    # Threshold is heuristic, depends on data generation.
    assert mean_cosine_similarity > 0.8, f"Mean cosine similarity to assigned centroid is low: {mean_cosine_similarity}"

# TODO: Add test for Euclidean metric to ensure it's not broken (optional for this specific subtask)

# Test for the bug fix related to device.type for pin_memory
def test_pin_memory_device_check(default_device_str):
    # This test is mostly to ensure the FastKMeans constructor runs with the fix
    # The actual pin_memory call happens during train()
    n_features, k = 10, 3
    data = generate_spherical_data(100, n_features, k)
    
    if CUDA_AVAILABLE and default_device_str == "cuda":
        model = FastKMeans(d=n_features, k=k, device="cuda", pin_gpu_memory=True)
        model.train(data) # This would have failed if device.type was not used
        assert True # If it runs, it's good
    else:
        model = FastKMeans(d=n_features, k=k, device="cpu", pin_gpu_memory=False)
        model.train(data)
        assert True


# Test to ensure Euclidean still works
euclidean_param_configs = []
euclidean_param_configs.append(pytest.param({"metric": "euclidean", "use_triton": False, "device_str": "cpu", "dtype": torch.float32}, id="euclidean-torch-cpu-fp32"))
if CUDA_AVAILABLE:
    euclidean_param_configs.append(pytest.param({"metric": "euclidean", "use_triton": False, "device_str": "cuda", "dtype": torch.float32}, id="euclidean-torch-gpu-fp32"))
    if HAS_TRITON:
        euclidean_param_configs.append(pytest.param({"metric": "euclidean", "use_triton": True, "device_str": "cuda", "dtype": torch.float32}, id="euclidean-triton-gpu-fp32"))


@pytest.mark.parametrize("config", euclidean_param_configs)
def test_euclidean_kmeans_basic(config, default_device_str):
    if config["use_triton"] and not CUDA_AVAILABLE:
        pytest.skip("Triton test requires CUDA.")
    if config["device_str"] == "cuda" and not CUDA_AVAILABLE:
        pytest.skip("CUDA test requires CUDA.")

    n_samples, n_features, k = 100, 10, 3
    # For Euclidean, data doesn't need to be spherical
    np.random.seed(42)
    data = np.random.rand(n_samples, n_features).astype(np.float32)


    model = FastKMeans(
        d=n_features,
        k=k,
        metric=config["metric"],
        use_triton=config["use_triton"],
        device=config["device_str"],
        dtype=config["dtype"],
        niter=5
    )
    model.train(data)

    assert model.centroids is not None
    assert model.centroids.shape == (k, n_features)
    # Euclidean centroids are not necessarily normalized
    centroid_norms = np.linalg.norm(model.centroids, axis=1)
    assert not np.allclose(centroid_norms, 1.0) or k==0 # Could be normalized by chance, but unlikely for general data

    labels = model.predict(data)
    assert labels.shape == (n_samples,)
    assert labels.min() >= 0
    assert labels.max() < k
