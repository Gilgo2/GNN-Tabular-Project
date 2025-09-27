from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

def test_model(model, test_data, device, is_gnn=False, batch_size=8, results_dir="results"):
    """
    Test a trained model and save metrics to a results file.

    Args:
        model      : trained DeepSets or GNN model
        test_data  : test dataset (list of sets or PyG graphs)
        device     : torch.device
        is_gnn     : whether the model is a GNN (True) or DeepSet (False)
        batch_size : batch size for evaluation
        results_dir: folder to store results json file
    """
    model.to(device)
    model.eval()

    if is_gnn:
        loader = PyGDataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        loader = TorchDataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda b: b)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            if is_gnn:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                labels = batch.y.view(-1)
            else:
                batch_logits, batch_labels = [], []
                for X_set, y in batch:
                    X_set = X_set.to(device)
                    logits = model(X_set)
                    batch_logits.append(logits.unsqueeze(0))
                    batch_labels.append(y.to(device).unsqueeze(0))
                logits = torch.cat(batch_logits, dim=0)
                labels = torch.cat(batch_labels, dim=0)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- metrics ---
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds).tolist()

    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    # --- save results ---
    import os
    os.makedirs(results_dir, exist_ok=True)
    class_name = model.__class__.__name__
    out_path = os.path.join(results_dir, f"{class_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved to {out_path}")
    return results
