import torch
from deepset import *
from data_gen import generate_deepset_data
from graph_gen import create_graph_dataset
from training import *
from GNNs import *
import json
from testing import *
from plotting import *

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    raw_data, labels = generate_deepset_data(5000, delta=2)
    graphs = create_graph_dataset(raw_data, labels, 4, device)

    splits = split_datasets(raw_data, labels, graphs, seed=42)

    X_train, y_train, g_train = splits['train']
    X_val, y_val, g_val = splits['val']
    X_test, y_test, g_test = splits['test']

    train_dataset = DeepSetsDataset(X_train, y_train, device=device)
    val_dataset = DeepSetsDataset(X_val, y_val, device=device)
    test_dataset = DeepSetsDataset(X_test, y_test, device=device)


    deepset_grid = {
        'hidden_dim': [32, 64, 128, 256, 512],
        'lr': [5e-3,1e-3, 1e-2],
        'batch_size': [8],
        'epochs': [50]
    }

    # GNNs
    gnn_grid = {
        'hidden_dim': [32, 64, 128, 256, 512],
        'lr': [5e-3,1e-3, 1e-2],
        'batch_size': [8],
        'epochs': [50]
    }

    deepset_best_model, deepset_best_params = grid_search_deepset(
        model_class=DeepSets,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        param_grid=deepset_grid,
        device=device
    )

    torch.save(deepset_best_model.state_dict(), 'deepset/deepset_best.pt')

    # Save hyperparameters to JSON
    with open('deepset/deepset_best_params.json', 'w') as f:
        json.dump(deepset_best_params, f)
    test_model(deepset_best_model, test_dataset, device=device, is_gnn=False, results_dir='deepset/results')

    gnn_models = [GCNNet, GraphSAGENet, GINNet, GATNet]
    gnn_model_names = ["GCN", "GraphSAGE", "GIN", "GAT"]
    for model_type, name in zip(gnn_models, gnn_model_names):
        gnn_best_model, gnn_best_params = grid_search_gnn(
            model_class=model_type,
            train_graphs=g_train,
            val_graphs=g_val,
            param_grid=gnn_grid,
            device=device
        )
        # Similarly for GNN
        torch.save(gnn_best_model.state_dict(), f"deepset/{name}.pt")
        with open(f"deepset/{name}_params.json", 'w') as f:
            json.dump(gnn_best_params, f)
        test_model(gnn_best_model, g_test, device=device, is_gnn=True, results_dir='deepset/results')



if __name__ == '__main__':
    main()
