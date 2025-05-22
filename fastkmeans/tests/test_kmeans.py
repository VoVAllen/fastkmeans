import pytest
import torch
import numpy as np
from fastkmeans.kmeans import FastKMeans, _kmeans_plusplus_init

# Helper function to create synthetic data
def create_synthetic_data(n_samples_per_cluster: int, n_features: int, n_clusters: int, separation: float, seed: int = 0):
    """Creates synthetic data with well-separated clusters."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    data_list = []
    cluster_ids_list = []
    
    for i in range(n_clusters):
        # Create a cluster center
        center = torch.randn(n_features) * separation * i
        
        # Generate points around the center
        points = torch.randn(n_samples_per_cluster, n_features) + center
        data_list.append(points)
        cluster_ids_list.extend([i] * n_samples_per_cluster)
        
    data = torch.cat(data_list, dim=0)
    cluster_ids = torch.tensor(cluster_ids_list, dtype=torch.long)
    
    return data, cluster_ids

# Test for _kmeans_plusplus_init
def test_kmeans_plusplus_init_well_separated_clusters():
    """
    Tests if _kmeans_plusplus_init picks one centroid from each well-separated cluster.
    """
    n_samples_per_cluster = 10
    n_features = 2
    n_clusters = 3
    separation = 10.0
    seed = 42
    device = torch.device("cpu")

    data, original_cluster_ids = create_synthetic_data(
        n_samples_per_cluster, n_features, n_clusters, separation, seed
    )
    data = data.to(device)

    centroids = _kmeans_plusplus_init(data, n_clusters, device, seed)

    assert centroids.shape == (n_clusters, n_features), "Centroids shape mismatch."
    assert centroids.device == device, "Centroids device mismatch."
    
    # Check if all returned centroids are from the original data points
    # This can be tricky due to floating point precision.
    # A simpler check is if they are distinct and their number is correct.
    # More robust: check if each centroid is "close" to some point in data.
    # For this test, since data points are well separated, chosen centroids should be exact points.
    
    # Check that k distinct points from the original dataset are returned
    # Convert to a set of tuples for uniqueness check
    data_tuples = {tuple(p.tolist()) for p in data}
    centroid_tuples = {tuple(c.tolist()) for c in centroids}
    
    assert len(centroid_tuples) == n_clusters, "Number of unique centroids is not k."
    for ct in centroid_tuples:
        assert ct in data_tuples, f"Centroid {ct} not found in original data."

    # Check if each centroid belongs to a different original cluster
    selected_centroid_original_cluster_ids = []
    for i in range(centroids.shape[0]):
        centroid = centroids[i]
        # Find which original data point this centroid corresponds to
        # This assumes centroids are exact copies of data points
        for j in range(data.shape[0]):
            if torch.equal(centroid, data[j]):
                selected_centroid_original_cluster_ids.append(original_cluster_ids[j].item())
                break
    
    assert len(set(selected_centroid_original_cluster_ids)) == n_clusters, \
        "Centroids were not picked from distinct original clusters."

def test_kmeans_plusplus_init_edge_cases():
    """Tests _kmeans_plusplus_init with edge cases."""
    n_features = 2
    seed = 42
    device = torch.device("cpu")

    # 1. Test with k=1
    data_k1, _ = create_synthetic_data(10, n_features, 1, 1.0, seed)
    data_k1 = data_k1.to(device)
    centroids_k1 = _kmeans_plusplus_init(data_k1, 1, device, seed)
    assert centroids_k1.shape == (1, n_features), "k=1: Centroids shape mismatch."
    assert tuple(centroids_k1[0].tolist()) in {tuple(p.tolist()) for p in data_k1}, "k=1: Centroid not from data."

    # 2. Test with data where all points are identical
    identical_points = torch.ones(20, n_features).to(device)
    k_identical = 3
    centroids_identical = _kmeans_plusplus_init(identical_points, k_identical, device, seed)
    assert centroids_identical.shape == (k_identical, n_features), "Identical points: Centroids shape mismatch."
    # All centroids should be identical to the input points
    for i in range(k_identical):
        assert torch.equal(centroids_identical[i], identical_points[0]), \
            f"Identical points: Centroid {i} is not the identical point."
    
    # 3. Test with k greater than the number of unique points
    unique_points_data, _ = create_synthetic_data(1, n_features, 2, 10.0, seed) # 2 unique points
    unique_points_data = unique_points_data.to(device)
    k_gt_unique = 3
    centroids_gt_unique = _kmeans_plusplus_init(unique_points_data, k_gt_unique, device, seed)
    assert centroids_gt_unique.shape == (k_gt_unique, n_features), "k > unique_points: Centroids shape mismatch."
    # Check that the returned centroids are from the original unique points
    # Some points will be duplicated among centroids in this case.
    unique_points_set = {tuple(p.tolist()) for p in unique_points_data}
    for i in range(k_gt_unique):
        assert tuple(centroids_gt_unique[i].tolist()) in unique_points_set, \
             f"k > unique_points: Centroid {i} not from original unique points."

# Test for FastKMeans class with different init methods
def test_fastkmeans_init_methods():
    """Tests FastKMeans with 'random' and 'kmeans++' initializations, and invalid init."""
    n_samples = 100
    n_features = 5
    k = 5
    seed = 42
    
    # Using numpy data as FastKMeans expects numpy array
    rng = np.random.RandomState(seed)
    data_np = rng.rand(n_samples, n_features).astype(np.float32)

    # Test with init='kmeans++'
    kmeans_plusplus = FastKMeans(d=n_features, k=k, seed=seed, init='kmeans++')
    kmeans_plusplus.train(data_np)
    assert kmeans_plusplus.centroids is not None, "kmeans++: Centroids not populated."
    assert kmeans_plusplus.centroids.shape == (k, n_features), "kmeans++: Centroids shape mismatch."
    centroids_pp = kmeans_plusplus.centroids.copy()

    # Test with init='random'
    kmeans_random = FastKMeans(d=n_features, k=k, seed=seed, init='random')
    kmeans_random.train(data_np)
    assert kmeans_random.centroids is not None, "random: Centroids not populated."
    assert kmeans_random.centroids.shape == (k, n_features), "random: Centroids shape mismatch."
    centroids_rand = kmeans_random.centroids.copy()

    # For a sufficiently complex dataset and same seed, different init methods should ideally lead to
    # different sets of initial centroids.
    # However, the core test is that they run and produce valid outputs.
    # A direct comparison of centroids might be flaky if k is very small or data is simple.
    # We'll check if they are different, assuming the dataset is complex enough.
    # If k=1, they might be the same if kmeans++ picks the same first random point.
    if k > 1:
         # It's possible but highly improbable they are identical for k > 1 with different logic paths
        assert not np.allclose(centroids_pp, centroids_rand), \
            "kmeans++ and random initializations produced identical centroids with the same seed for k > 1."

    # Test invalid init method
    with pytest.raises(ValueError, match="Invalid init method: invalid_method. Valid options are 'random' and 'kmeans++'."):
        FastKMeans(d=n_features, k=k, seed=seed, init='invalid_method')

def test_fastkmeans_predict_after_train():
    """Tests if predict runs after training."""
    n_samples = 30
    n_features = 3
    k = 3
    seed = 42
    rng = np.random.RandomState(seed)
    data_np = rng.rand(n_samples, n_features).astype(np.float32)

    kmeans = FastKMeans(d=n_features, k=k, seed=seed, init='kmeans++')
    kmeans.train(data_np)
    labels = kmeans.predict(data_np)
    assert labels is not None
    assert labels.shape == (n_samples,)
    assert len(np.unique(labels)) <= k

    kmeans_random = FastKMeans(d=n_features, k=k, seed=seed, init='random')
    kmeans_random.train(data_np)
    labels_random = kmeans_random.predict(data_np)
    assert labels_random is not None
    assert labels_random.shape == (n_samples,)
    assert len(np.unique(labels_random)) <= k

# It might be good to add a test for GPU usage if possible,
# but that often requires specific hardware and setup.
# For now, tests focus on CPU behavior.

# Example of how to run (if this were a script, not for the agent to run directly):
# if __name__ == "__main__":
#     pytest.main([__file__])
print("Test file created: fastkmeans/tests/test_kmeans.py")
