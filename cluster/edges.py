import torch
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

def build_graph(X, knn_k=20, p=None, r=None, settings='knn'):
    """Build KNN graph structure for GNN."""
    if settings == 'knn' and knn_k > 0:
        edge_index = create_knn_graph(X, knn_k=knn_k)
    elif settings == 'random':
        edge_index = create_erdos_renyi_edges(len(X), p=p, seed=42)
    elif settings == 'regular':
        edge_index = create_random_regular_edges(len(X), degree=r, seed=42)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
    return edge_index

def create_knn_graph(X, knn_k=20):
    neigh = NearestNeighbors(n_neighbors=knn_k)
    neigh.fit(X)
    knn_graph = neigh.kneighbors_graph(X, mode='connectivity')
    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long)
    return edge_index

def create_erdos_renyi_edges(num_nodes, p, seed=None):
    """
    Create an Erdos-Renyi random graph edge_index.
    Args:
        num_nodes (int): Number of nodes in the graph.
        p (float): Probability of edge creation between any two nodes.
        seed (int, optional): Random seed for reproducibility.
    Returns:
        edge_index (torch.LongTensor): Edge index tensor of shape [2, num_edges].
    """
    import torch
    import numpy as np
    rng = np.random.default_rng(seed)
    edge_set = set()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < p:
                edge_set.add((i, j))
                edge_set.add((j, i))
    if edge_set:
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index

def create_random_regular_edges(num_nodes, degree, seed=None):
    """
    Create a random regular graph edge_index where each node has the same degree.
    Args:
        num_nodes (int): Number of nodes in the graph.
        degree (int): Degree of each node (must be even if num_nodes is odd).
        seed (int, optional): Random seed for reproducibility.
    Returns:
        edge_index (torch.LongTensor): Edge index tensor of shape [2, num_edges].
    """
    import torch
    import networkx as nx
    if degree >= num_nodes:
        raise ValueError("Degree must be less than the number of nodes.")
    if (num_nodes * degree) % 2 != 0:
        raise ValueError("num_nodes * degree must be even for a regular graph.")
    G = nx.random_regular_graph(degree, num_nodes, seed=seed)
    edge_set = set()
    for u, v in G.edges():
        edge_set.add((u, v))
        edge_set.add((v, u))
    if edge_set:
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index