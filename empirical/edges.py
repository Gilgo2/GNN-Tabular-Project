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
        print(len(edge_set), "edges created using Erdos-Renyi model with p =", p)
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

def create_knn_edges(df, num_top_features, dataset_name, k_neighbors=5):
    # Only use kNN on the top features (numerical or categorical, but treat all as numeric for kNN)
    import pandas as pd
    import torch
    from sklearn.neighbors import NearestNeighbors

    with open(f'empirical/runs/{dataset_name}/feature_importances.csv', 'r') as f:
        feature_importances = pd.read_csv(f)
    top_features = feature_importances['feature'][:num_top_features].tolist()
    top_features = [f.replace('num__', '') for f in top_features]
    
    # print("Top features used for kNN edge creation:", top_features)

    if not top_features:
        print("No top features selected, returning empty edge_index.")
        return torch.empty((2, 0), dtype=torch.long)
    if top_features[0][0] == 'x' and top_features[0][1:].isdigit():
        top_features = [int(feature[1:]) for feature in top_features]
    X_top = df[top_features].values
    n_neighbors = min(k_neighbors + 1, len(df))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X_top)
    distances, indices = nbrs.kneighbors(X_top)

    edge_set = set()
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self (first neighbor)
            edge_set.add((i, j))
            edge_set.add((j, i))
    print(len(edge_set), "edges created using kNN on top features.")
    if edge_set:
        edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index

def create_edges(df, num_top_features, dataset_name, k=None, p=None, r=None, seed=None):
    """Create edge_index based on specified method."""
    import torch
    num_nodes = df.shape[0]
    if k is not None:
        edge_index = create_knn_edges(df, num_top_features, dataset_name, k_neighbors=k)
    elif p is not None:
        edge_index = create_erdos_renyi_edges(num_nodes, p, seed=seed)
    elif r is not None:
        edge_index = create_random_regular_edges(num_nodes, r, seed=seed)
    else:
        raise ValueError("One of k, p, or r must be specified for edge creation.")
    return edge_index