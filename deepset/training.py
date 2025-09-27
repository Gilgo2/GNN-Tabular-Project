import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

import numpy as np
import itertools


def grid_search(model_class, dataset_train, dataset_val, param_grid, device, is_gnn=False):
    best_acc = 0
    best_params = None
    best_model = None

    # create all combinations
    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"Testing params: {params}")

        # create model
        if is_gnn:
            model = model_class(hidden_dim=params['hidden_dim'])
            train_gnn(model, dataset_train, dataset_val,
                               epochs=params['epochs'], batch_size=params['batch_size'],
                               lr=params['lr'], device=device)
        else:
            model = model_class(hidden_dim=params['hidden_dim'])
            train_deepset(model, dataset_train, dataset_val,
                                   epochs=params['epochs'], batch_size=params['batch_size'],
                                   lr=params['lr'], device=device)


def split_datasets(sets, labels, graphs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=None):
    """
    Split DeepSets and Graph datasets into train/val/test in the same way.

    Parameters:
        sets      : list of arrays (DeepSets)
        labels    : array of dataset-level labels
        graphs    : list of graph dicts (edges, X, y)
        train_ratio, val_ratio, test_ratio : split ratios (must sum to 1)
        seed      : random seed

    Returns:
        dict with keys: 'train', 'val', 'test'
        Each is a tuple: (sets_split, labels_split, graphs_split)
    """
    assert len(sets) == len(labels) == len(graphs), "Datasets must have same length"
    rng = np.random.default_rng(seed)
    n = len(sets)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    def split(idx):
        return ([sets[i] for i in idx],
                labels[idx],
                [graphs[i] for i in idx])

    return {
        'train': split(train_idx),
        'val': split(val_idx),
        'test': split(test_idx)
    }

def train_deepset(model, train_dataset, val_dataset, device, epochs=20, batch_size=8, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # decay LR every 5 epochs

    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch
    )

    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch
    )

    for epoch in range(1, epochs+1):
        # --- training ---
        model.train()
        total_correct, total_samples = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_logits, batch_labels = [], []
            for X_set, y in batch:
                X_set = X_set.to(device)
                logits = model(X_set)
                batch_logits.append(logits.unsqueeze(0))
                batch_labels.append(y.to(device).unsqueeze(0))
            logits = torch.cat(batch_logits, dim=0)
            labels = torch.cat(batch_labels, dim=0)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()  # update LR

        # --- validation ---
        model.eval()
        val_correct, val_samples = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch_logits, batch_labels = [], []
                for X_set, y in batch:
                    X_set = X_set.to(device)
                    logits = model(X_set)
                    batch_logits.append(logits.unsqueeze(0))
                    batch_labels.append(y.to(device).unsqueeze(0))
                logits = torch.cat(batch_logits, dim=0)
                labels = torch.cat(batch_labels, dim=0)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_samples += labels.size(0)

        print(f"[DeepSets] Epoch {epoch:02d}: Train Acc={total_correct/total_samples:.4f}, Val Acc={val_correct/val_samples:.4f}, LR={scheduler.get_last_lr()[0]:.5f}")


def train_gnn(model, train_graphs, val_graphs, device, epochs=20, batch_size=8, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    train_loader = PyGDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = PyGDataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs+1):
        # --- training ---
        model.train()
        total_correct, total_samples = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.view(-1)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        # --- validation ---
        model.eval()
        val_correct, val_samples = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.view(-1)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_samples += labels.size(0)

        train_acc = total_correct / total_samples
        val_acc = val_correct / val_samples
        print(f"[GNN] Epoch {epoch:02d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, LR={scheduler.get_last_lr()[0]:.5f}")




def grid_search_deepset(model_class, train_dataset, val_dataset, param_grid, device):
    import itertools
    best_acc = 0
    best_model = None
    best_params = None

    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"Testing params: {params}")

        model = model_class(hidden_dim=params['hidden_dim']).to(device)

        # train model
        train_deepset(model, train_dataset, val_dataset,
                               epochs=params['epochs'],
                               batch_size=params['batch_size'],
                               lr=params['lr'],
                               device=device)

        # compute validation accuracy
        val_correct, val_total = 0, 0
        model.eval()
        with torch.no_grad():
            for X_set, y in val_dataset:
                logits = model(X_set.to(device))
                pred = logits.argmax(dim=-1)
                val_correct += (pred == y.to(device)).sum().item()
                val_total += 1
        val_acc = val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_params = params

    print("Best DeepSets params:", best_params, "Val Acc:", best_acc)
    return best_model, best_params

def grid_search_gnn(model_class, train_graphs, val_graphs, param_grid, device):
    import itertools
    best_acc = 0
    best_model = None
    best_params = None

    keys, values = zip(*param_grid.items())
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        print(f"Testing params: {params}")

        model = model_class(hidden_dim=params['hidden_dim']).to(device)

        # Use the corrected train_gnn with PyGDataLoader
        train_gnn(model, train_graphs, val_graphs,
                  epochs=params['epochs'],
                  batch_size=params['batch_size'],
                  lr=params['lr'],
                  device=device)

        # Validation accuracy using PyG DataLoader
        val_loader = PyGDataLoader(val_graphs, batch_size=params['batch_size'], shuffle=False)
        val_correct, val_total = 0, 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                pred = logits.argmax(dim=-1)
                val_correct += (pred == batch.y.view(-1)).sum().item()
                val_total += batch.y.size(0)

        val_acc = val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_params = params

    print("Best GNN params:", best_params, "Val Acc:", best_acc)
    return best_model, best_params