import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_point_sets(point_sets, labels, max_dim=2):
    """
    Plot multiple sets of points in d-dim space.

    Args:
        point_sets: list of np.arrays, each of shape (n_points, d)
        labels: list of labels, one per point set
        max_dim: 2 or 3, for 2D or 3D plotting
    """
    # Concatenate for PCA if needed
    all_points = np.vstack(point_sets)
    d = all_points.shape[1]

    # Reduce dimensionality if d > max_dim
    if d > max_dim:
        reducer = PCA(n_components=max_dim)
        all_points = reducer.fit_transform(all_points)

        # split back
        split_sizes = [ps.shape[0] for ps in point_sets]
        point_sets_proj = []
        idx = 0
        for size in split_sizes:
            point_sets_proj.append(all_points[idx:idx+size])
            idx += size
    else:
        point_sets_proj = point_sets

    # Plot
    fig = plt.figure()
    title = "PCA to 2d of set points colored by label"

    if max_dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        for ps, label in zip(point_sets_proj, labels):
            ax.scatter(ps[:,0], ps[:,1], ps[:,2], label=label, alpha=0.7)
    else:
        plt.title(title)
        for ps, label in zip(point_sets_proj, labels):
            plt.scatter(ps[:,0], ps[:,1], label=label, alpha=0.7)

    plt.legend()
    plt.show()