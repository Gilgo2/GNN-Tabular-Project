import numpy as np
import torch
from torch_geometric.data import Data

def point_set_to_graph_object(points, label, k, device):
    """
    Convert a point set into a PyTorch Geometric Data graph object with a graph-level label.

    Args:
        points: np.array of shape (n_points, d) - node features
        label: int - graph-level target
        k: number of nearest neighbors per node
        device: torch.device to place tensors on

    Returns:
        graph: torch_geometric.data.Data object with x, edge_index, and y on the specified device
    """
    n_points = points.shape[0]

    # Compute pairwise distances
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)  # remove self-loops

    # k nearest neighbors
    neighbors = np.argsort(dists, axis=1)[:, :k]

    # Build edge index (directed)
    src = np.repeat(np.arange(n_points), k)
    tgt = neighbors.flatten()
    edge_index = torch.tensor(np.stack([src, tgt], axis=0), dtype=torch.long, device=device)

    # Node features
    x = torch.tensor(points, dtype=torch.float, device=device)

    # Graph-level label
    y = torch.tensor([label], dtype=torch.long, device=device)

    # Create PyG Data object
    graph = Data(x=x, edge_index=edge_index, y=y)
    return graph


def create_graph_dataset(data_sets, labels, k, device):
    """
    Convert a list of point sets and their labels into a list of PyG Data objects on a specified device.

    Args:
        data_sets: list of np.array, each shape (n_points, d)
        labels: list of int, graph-level labels
        k: int, number of nearest neighbors for k-NN graph
        device: torch.device to place tensors on

    Returns:
        graph_list: list of PyG Data objects with x, edge_index, and y on the specified device
    """
    graph_list = []

    for points, label in zip(data_sets, labels):
        graph = point_set_to_graph_object(points, label, k, device)
        graph_list.append(graph)

    return graph_list


