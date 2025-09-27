import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class DeepSetsDataset(Dataset):
    """
    PyTorch dataset for DeepSets. Each sample is a set of points with a label.
    """
    def __init__(self, X_sets, y_labels, device):
        """
        Parameters:
            X_sets     : list of arrays, each (n_points_i, d)
            y_labels   : array of dataset-level labels
            device     : 'cpu' or 'cuda'
        """
        assert len(X_sets) == len(y_labels)
        self.X_sets = [torch.tensor(x, dtype=torch.float32, device=device) for x in X_sets]
        self.y_labels = torch.tensor(y_labels, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.X_sets)

    def __getitem__(self, idx):
        return self.X_sets[idx], self.y_labels[idx]


class DeepSets(nn.Module):
    """
    Minimal DeepSets model for set-level classification.
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        """
        Parameters:
            input_dim  : dimensionality of each element in the set (here 2)
            hidden_dim : hidden layer size in phi and rho
            output_dim : number of classes (here 3: blobs, circles, moons)
        """
        super().__init__()
        # phi: element-wise embedding
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # rho: set-level embedding to output
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_set):
        h = self.phi(x_set)  # (n_points, hidden_dim)
        h_agg = h.mean(dim=0)  # mean aggregation instead of sum
        out = self.rho(h_agg)  # (output_dim,)
        return out
