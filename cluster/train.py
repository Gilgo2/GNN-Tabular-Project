from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import wandb
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from edges import build_graph
import torch
from sklearn.neighbors import NearestNeighbors
from models import CONVRegressor, CONVClassifier
import torch.nn.functional as F
import numpy as np
from data_utils import get_small_cluster_labels
import os

def train_ml_model(data_splits, model_type='mlp', learning_rate=0.001, max_iter=1000, random_state=42):
    """Train traditional ML model and log to wandb."""

    if model_type.lower() == 'mlp':
        # MLP Regressor
        reg_model = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=random_state
        )
        reg_model.fit(data_splits['X_train'], data_splits['y_reg_train'])

        # MLP Classifier
        class_model = MLPClassifier(
            hidden_layer_sizes=(128, 128),
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=random_state
        )
        class_model.fit(data_splits['X_train'], data_splits['y_class_train'])

        # Log MLP training curves (loss values from model)
        if hasattr(reg_model, 'loss_curve_'):
            for epoch, loss in enumerate(reg_model.loss_curve_):
                wandb.log({"regression_train_loss": loss, "epoch": epoch})

        if hasattr(class_model, 'loss_curve_'):
            for epoch, loss in enumerate(class_model.loss_curve_):
                wandb.log({"classification_train_loss": loss, "epoch": epoch})

    elif model_type.lower() == 'rf':
        # Random Forest models
        reg_model = RandomForestRegressor(
            n_estimators=max_iter//10,  # Use max_iter as proxy for n_estimators
            random_state=random_state
        )
        reg_model.fit(data_splits['X_train'], data_splits['y_reg_train'])

        class_model = RandomForestClassifier(
            n_estimators=max_iter//10,
            random_state=random_state
        )
        class_model.fit(data_splits['X_train'], data_splits['y_class_train'])

        # RF doesn't have loss curves, so log final metrics only
        wandb.log({"model_trained": True})

    # Evaluate models
    reg_pred_train = reg_model.predict(data_splits['X_train'])
    reg_pred_val = reg_model.predict(data_splits['X_val'])
    reg_pred_test = reg_model.predict(data_splits['X_test'])

    class_pred_train = class_model.predict(data_splits['X_train'])
    class_pred_val = class_model.predict(data_splits['X_val'])
    class_pred_test = class_model.predict(data_splits['X_test'])

    small_cluster_labels = get_small_cluster_labels(data_splits)
    # Calculate metrics
    results = {
        'regression': {
            'train_mae': mean_absolute_error(data_splits['y_reg_train'], reg_pred_train),
            'val_mae': mean_absolute_error(data_splits['y_reg_val'], reg_pred_val),
            'test_mae': mean_absolute_error(data_splits['y_reg_test'], reg_pred_test),
            'train_r2': r2_score(data_splits['y_reg_train'], reg_pred_train),
            'val_r2': r2_score(data_splits['y_reg_val'], reg_pred_val),
            'test_r2': r2_score(data_splits['y_reg_test'], reg_pred_test),
        },
        'classification': {
            'train_accuracy': accuracy_score(data_splits['y_class_train'], class_pred_train),
            'val_accuracy': accuracy_score(data_splits['y_class_val'], class_pred_val),
            'test_accuracy': accuracy_score(data_splits['y_class_test'], class_pred_test)
        }
    }
    for size, index in small_cluster_labels.items():
        results['regression'][f'{size}_cluster_mae'] = mean_absolute_error(
            [data_splits['y_reg_test'][index]],
            [reg_pred_test[index]]
        )
        results['classification'][f'{size}_cluster_accuracy'] = accuracy_score(
            [data_splits['y_class_test'][index]],
            [class_pred_test[index]]
        )

    return results

def train_gnn_model(X, y_reg, y_class, data_splits, knn, n_features, n_classes, settings='knn',
                   learning_rate=0.001, iterations=1000, p=None, r=None, cluster_distance_multiplier=1):
    """Train GNN model and log to wandb."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    edge_index = build_graph(X, knn_k=knn, settings=settings, p=p, r=r)

    # Create masks
    train_mask = torch.zeros(len(y_reg), dtype=torch.bool)
    val_mask = torch.zeros(len(y_reg), dtype=torch.bool)
    test_mask = torch.zeros(len(y_reg), dtype=torch.bool)
    train_mask[data_splits['train_idx']] = True
    val_mask[data_splits['val_idx']] = True
    test_mask[data_splits['test_idx']] = True

    # Prepare data
    features = torch.tensor(X, dtype=torch.float).to(device)
    reg_labels = torch.tensor(y_reg, dtype=torch.float).to(device)
    class_labels = torch.tensor(y_class, dtype=torch.long).to(device)
    edge_index = edge_index.to(device)

    # Initialize models
    reg_model = CONVRegressor(n_features, 128, 1, num_layers=3).to(device)
    class_model = CONVClassifier(n_features, 128, n_classes, num_layers=3).to(device)

    # Training regression model
    reg_optimizer = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)
    reg_model.train()

    for epoch in range(iterations):
        reg_optimizer.zero_grad()
        out = reg_model(features, edge_index)
        train_loss = F.mse_loss(out[train_mask], reg_labels[train_mask])
        train_loss.backward()
        reg_optimizer.step()

        # Validation loss
        if epoch % 50 == 0:
            reg_model.eval()
            with torch.no_grad():
                val_out = reg_model(features, edge_index)
                val_loss = F.mse_loss(val_out[val_mask], reg_labels[val_mask])
            reg_model.train()

            wandb.log({
                "regression_train_loss": train_loss.item(),
                "regression_val_loss": val_loss.item(),
                "epoch": epoch
            })

    # Training classification model
    class_optimizer = torch.optim.SGD(class_model.parameters(), lr=learning_rate)
    class_model.train()

    for epoch in range(iterations):
        class_optimizer.zero_grad()
        out = class_model(features, edge_index)
        train_loss = F.nll_loss(out[train_mask], class_labels[train_mask])
        train_loss.backward()
        class_optimizer.step()

        # Validation loss
        if epoch % 50 == 0:
            class_model.eval()
            with torch.no_grad():
                val_out = class_model(features, edge_index)
                val_loss = F.nll_loss(val_out[val_mask], class_labels[val_mask])
            class_model.train()

            wandb.log({
                "classification_train_loss": train_loss.item(),
                "classification_val_loss": val_loss.item(),
                "epoch": epoch
            })

    # Final evaluation
    reg_model.eval()
    class_model.eval()

    with torch.no_grad():
        reg_pred = reg_model(features, edge_index).cpu().numpy()
        class_pred = class_model(features, edge_index).cpu().argmax(dim=1).numpy()

    small_cluster_labels = get_small_cluster_labels(data_splits)
    # Calculate metrics
    results = {
        'regression': {
            'train_mae': mean_absolute_error(y_reg[data_splits['train_idx']], reg_pred[data_splits['train_idx']]),
            'val_mae': mean_absolute_error(y_reg[data_splits['val_idx']], reg_pred[data_splits['val_idx']]),
            'test_mae': mean_absolute_error(y_reg[data_splits['test_idx']], reg_pred[data_splits['test_idx']]),
            'train_r2': r2_score(y_reg[data_splits['train_idx']], reg_pred[data_splits['train_idx']]),
            'val_r2': r2_score(y_reg[data_splits['val_idx']], reg_pred[data_splits['val_idx']]),
            'test_r2': r2_score(y_reg[data_splits['test_idx']], reg_pred[data_splits['test_idx']])
        },
        'classification': {
            'train_accuracy': accuracy_score(y_class[data_splits['train_idx']], class_pred[data_splits['train_idx']]),
            'val_accuracy': accuracy_score(y_class[data_splits['val_idx']], class_pred[data_splits['val_idx']]),
            'test_accuracy': accuracy_score(y_class[data_splits['test_idx']], class_pred[data_splits['test_idx']])
        }
    }
    for size, index in small_cluster_labels.items():
        results['regression'][f'{size}_cluster_mae'] = mean_absolute_error(
            [y_reg[data_splits['test_idx']][index]],
            [reg_pred[data_splits['test_idx']][index]]
        )
        results['classification'][f'{size}_cluster_accuracy'] = accuracy_score(
            [y_class[data_splits['test_idx']][index]],
            [class_pred[data_splits['test_idx']][index]]
        )
    def save_model(model, p, k, r, output_name):
   
        if k:
            output_name += f'_k={k}.pth'
        elif p:
            output_name += f'_p={p}.pth'
        else:
            output_name += f'_r={r}.pth'
        torch.save(model.state_dict(), output_name)

    if not os.path.exists('cluster/runs'):
        os.mkdir('cluster/runs')
    save_model(reg_model, p, knn, r, f'cluster/runs/reg_conv_128x128_iterations={iterations}_lr={learning_rate}_clusterd={cluster_distance_multiplier}')
    save_model(class_model, p, knn, r, f'cluster/runs/class_conv_128x128_iterations={iterations}_lr={learning_rate}_clusterd={cluster_distance_multiplier}')
    return results
    return results