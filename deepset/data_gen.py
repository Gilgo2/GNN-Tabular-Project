import numpy as np
from torch_geometric.data import Data
from sklearn.datasets import make_blobs, make_circles, make_moons
import torch




def generate_point_set(n_points, delta):
    d = 3
    distributions = ["normal", "uniform", "exponential", "laplace", "beta", "gamma"]
    points = []

    # pick a single distribution for the set
    choice = np.random.choice(distributions)
    for _ in range(n_points):
        if choice == "normal":
            point = np.random.normal(loc=0, scale=1, size=d)
        elif choice == "uniform":
            point = np.random.uniform(low=-1, high=1, size=d)
        elif choice == "exponential":
            point = np.random.exponential(scale=1, size=d)
        elif choice == "laplace":
            point = np.random.laplace(loc=0, scale=1, size=d)
        elif choice == "beta":
            point = np.random.beta(a=2, b=5, size=d)
        elif choice == "gamma":
            point = np.random.gamma(shape=2, scale=2, size=d)
        points.append(point)

    points = np.array(points)

    # Compute per-point contributions based on neighbors
    contributions = []
    for i, p in enumerate(points):
        distances = np.linalg.norm(points - p, axis=1)
        neighbor_mask = (distances <= delta) & (distances > 0)  # exclude self
        neighbors = points[neighbor_mask]

        if len(neighbors) == 0:
            contribution = np.zeros(d)  # no neighbors
        else:
            # Example: mean of neighbor vectors
            contribution = neighbors.mean(axis=0)
        contributions.append(contribution)

    contributions = np.array(contributions)

    # Aggregate contributions to form a single set-level vector
    # Options: mean, sum, or something else. Here we use mean
    aggregated_vector = contributions.mean(axis=0)

    return points, aggregated_vector


def generate_deepset_data(n_sets, delta, min_points=100, max_points=250, n_classes=3):
    """
    Generates a dataset of sets for DeepSets.
    Each set has a random number of points between min_points and max_points.
    Returns:
        data: list of np.arrays, each array shape (n_points, d)
        labels: np.array of integers, class label per set (quantile-based)
        aggregated_vectors: np.array of aggregated vectors per set
    """
    data = []
    aggregated_vectors = []

    # Step 1: generate sets and compute aggregated vectors
    for _ in range(n_sets):
        n_points = np.random.randint(min_points, max_points+1)
        points, agg_vec = generate_point_set(n_points, delta)  # now returns aggregated vector
        data.append(points)
        aggregated_vectors.append(agg_vec)

    aggregated_vectors = np.array(aggregated_vectors)

    # Step 2: assign class labels based on quantiles
    # compute edges for uniform class distribution
    quantiles = np.linspace(0, 1, n_classes+1)
    edges = np.quantile(aggregated_vectors.mean(axis=1), quantiles)  # aggregate to scalar per set
    labels = np.digitize(aggregated_vectors.mean(axis=1), edges[1:-1])

    return data, labels





