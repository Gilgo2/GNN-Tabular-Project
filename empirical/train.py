
import torch 
from models import GraphConvModel
import numpy as np
from sklearn.metrics import f1_score

def train_gnn(
    data,
    epochs: int = 1000,
    lr: float = 0.01,
    hid: int = 64,
    layers: int = 2,
    val_every: int = 10,
    batch_size: int = 32,
    dropout_rate: float = 0.5
):
    """
    Train a Graph Convolutional Model on the given data.

    Returns:
        model (torch.nn.Module): trained model (best state loaded)
        test_f1 (float): test F1 score of the best model
        best_epoch (int): epoch at which the best model occurred
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Setup model, optimizer, criterion ----
    model = GraphConvModel(
        in_channels=data.num_features,
        hidden_dim=hid,
        num_classes=len(data.y.unique()),
        num_layers=layers).to(device)

    data = data.to(device)
    criterion = _build_loss_with_class_weights(data, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Training state tracking ----
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0

    train_indices = data.train_mask.nonzero(as_tuple=True)[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_indices.size(0), device=device)
        shuffled = train_indices[perm]

        for i in range(0, shuffled.size(0), batch_size):
            batch_idx = shuffled[i:i + batch_size]
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[batch_idx], data.y[batch_idx])
            loss.backward()
            optimizer.step()

        if epoch % val_every == 0:
            val_loss = _evaluate_loss(model, data, criterion, mask=data.val_mask)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                best_epoch = epoch

    # ---- Load best model & final evaluation ----
    if best_state is not None:
        model.load_state_dict(best_state)

    val_f1, test_f1 = _evaluate_f1(model, data)
    print(f'GNN F1 Score (val): {val_f1:.4f}')
    print(f'GNN F1 Score (test): {test_f1:.4f}')

    return model, test_f1, best_epoch


# === Helper functions ===

def _build_loss_with_class_weights(data, device):
    """Compute class weights inversely proportional to frequency."""
    y_train = data.y[data.train_mask].cpu().numpy()
    classes = np.unique(y_train)
    class_counts = np.array([(y_train == c).sum() for c in classes])
    class_weights = class_counts.sum() / (len(classes) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    return torch.nn.CrossEntropyLoss(weight=class_weights)


def _evaluate_loss(model, data, criterion, mask):
    """Evaluate loss on a specific mask."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        return criterion(out[mask], data.y[mask]).item()


def _evaluate_f1(model, data):
    """Compute F1 scores on val and test sets."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        _, pred = logits.max(dim=1)
        val_f1 = f1_score(
            data.y[data.val_mask].cpu().numpy(),
            pred[data.val_mask].cpu().numpy(),
            average='weighted'
        )
        test_f1 = f1_score(
            data.y[data.test_mask].cpu().numpy(),
            pred[data.test_mask].cpu().numpy(),
            average='weighted'
        )
    return val_f1, test_f1
