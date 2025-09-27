import numpy as np
from typing import Tuple, Dict
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
EXTRA_CLUSTERS = [5, 10, 15]
def split_data(X: np.ndarray, y_reg: np.ndarray, y_class: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Split data into train/val/test sets with stratification.
    """
    idx = np.arange(len(y_reg))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.3, random_state=random_state, stratify=y_class
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=random_state, stratify=y_class[temp_idx]
    )

    return {
        "X_train": X[train_idx], "X_val": X[val_idx], "X_test": X[test_idx],
        "y_reg_train": y_reg[train_idx], "y_reg_val": y_reg[val_idx], "y_reg_test": y_reg[test_idx],
        "y_class_train": y_class[train_idx], "y_class_val": y_class[val_idx], "y_class_test": y_class[test_idx],
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx
    }

def generate_data(
    n_communities: int = 30,
    nodes_per_community: int = 100,
    n_features: int = 4,
    random_state: int = 42,
    cluster_distance_multiplier: float = 6.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic community data with bias and save CSV + scatter plot.
    """
    np.random.seed(random_state)

    # Add extra clusters
    
    nodes_for_community = [nodes_per_community] * n_communities + EXTRA_CLUSTERS
    n_rows = sum(nodes_for_community)
    n_communities += len(EXTRA_CLUSTERS)

    # Community centers
    range_start, range_end = -1, 1
    cluster_distance = (range_end - range_start) / (n_communities * n_features)
    centers = np.arange(range_start, range_end, cluster_distance).reshape(n_communities, n_features)

    X = np.zeros((n_rows, n_features))
    community_ids = np.zeros(n_rows, dtype=int)

    start = 0
    for i, nodes in enumerate(nodes_for_community):
        end = start + nodes
        noise = np.random.uniform(
            -cluster_distance * cluster_distance_multiplier,
            cluster_distance * cluster_distance_multiplier,
            size=(nodes, n_features)
        )
        X[start:end] = centers[i] + noise
        community_ids[start:end] = i
        start = end

    # Bias & labels
    community_bias = np.random.uniform(-3, 3, size=n_communities)
    np.random.shuffle(community_bias)
    bias = community_bias[community_ids]

    X_scaled = StandardScaler().fit_transform(X)
    sums = X_scaled.sum(axis=1)
    y_regression = np.power(sums + bias, 3) / 20
    y_classification = community_ids

    # Save to CSV
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["y_classification"] = y_classification
    df["community_ids"] = community_ids
    df["bias"] = bias
    df["y_regression"] = y_regression
    os.makedirs("cluster/data", exist_ok=True)
    csv_path = f"cluster/data/dataset_{cluster_distance_multiplier}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset to {csv_path}")

    return X_scaled, y_regression, y_classification, community_ids, bias

def get_small_cluster_labels(data_splits):
    """
    Get indexes of nodes in smaller clusters for evaluation.
    """
    label_unique_count = np.concatenate([
        data_splits['y_class_train'],
        data_splits['y_class_val'],
        data_splits['y_class_test']
    ]).max().item() + 1
    indexes = np.arange(label_unique_count - len(EXTRA_CLUSTERS), label_unique_count)
    return {size: index for (size, index) in zip(EXTRA_CLUSTERS, indexes)}
    
    
