import os
import time
import itertools
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from edges import create_edges


# ============ PREPROCESSING ============

def build_preprocessor(df):
    """Build a preprocessor for numeric & categorical columns."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols + ['target']]

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols)
    ])

    return preprocessor, cat_cols, num_cols

def df_to_graph_data(df, num_top_features, dataset_name, seed, k=None, p=None, r=None):
    preprocessor, _, _ = build_preprocessor(df)
    X = torch.tensor(preprocessor.fit_transform(df.drop('target', axis=1)), dtype=torch.float)
    y = torch.tensor(df['target'].values, dtype=torch.long)
    edge_index = create_edges(df, num_top_features, dataset_name, k, p, r, seed)
    return Data(x=X, edge_index=edge_index, y=y)

def save_best_tabular_model_feature_importance(df, train_mask, val_mask, test_mask, dataset_name):
    """Train baseline ML models and save the best one."""
    preprocessor, cat_cols, _ = build_preprocessor(df)
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    best_score = -1
    best_model, best_name = None, None
    costs = []

    for name, model in models.items():
        pipe = Pipeline([('pre', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        cost = f1_score(y_val, preds, average='weighted')
        costs.append(cost)

        if cost > best_score:
            best_score, best_model, best_name = cost, pipe, name

    print_test_performance(best_model, X_test, y_test, best_name)
    save_feature_importances(best_model, cat_cols, dataset_name)


def print_test_performance(best_model, X_test, y_test, best_name):
    preds = best_model.predict(X_test)
    print(f"Best model {best_name} Test F1 Score: {f1_score(y_test, preds, average='weighted')}")


def save_feature_importances(best_model, cat_cols, dataset_name):
    feature_names = best_model.named_steps['pre'].get_feature_names_out()
    importances = best_model.named_steps['clf'].feature_importances_

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    grouped_names = [
        "_".join(name.split("_")[:-1]) if "_".join(name.split("_")[:-1]) in cat_cols else name
        for name in feat_imp.index
    ]
    feat_imp_grouped = feat_imp.groupby(grouped_names).sum().sort_values(ascending=False).reset_index()
    feat_imp_grouped.columns = ['feature', 'imp']

    print("Top Feature Importances:")
    print(feat_imp_grouped.head(20))
    feat_imp_grouped.to_csv(f'empirical/runs/{dataset_name}/feature_importances.csv', index=False)

def save_model(model, p, k, r, top, output_name):
    """Save GNN model state dict with proper naming."""
    if k:
        output_name += f'_k={k}_top={top}.pth'
    elif p:
        output_name += f'_p={p}.pth'
    else:
        output_name += f'_r={r}.pth'
    torch.save(model.state_dict(), output_name)


def set_data_masks(data, train_mask, val_mask, test_mask):
    """Convert masks to torch tensors and attach to data object."""
    num_nodes = data.num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_mask] = True
    data.val_mask[val_mask] = True
    data.test_mask[test_mask] = True
    return data


def did_run_already(result_df, num_top_features, hid, layers, lr, epochs, k, p, r, seed):
    """Check if hyperparameter configuration already ran."""
    if result_df.empty:
        return False

    existing = result_df[
        ((result_df['num_top_features'] == num_top_features) | (result_df['num_top_features'].isna())) &
        (result_df['hid'] == hid) &
        (result_df['layers'] == layers) &
        (result_df['lr'] == lr) &
        (result_df['epoch'] == epochs) &
        ((result_df['k'] == k) | (result_df['k'].isna())) &
        ((result_df['p'] == p) | (result_df['p'].isna())) &
        ((result_df['r'] == r) | (result_df['r'].isna())) &
        (result_df['seed'] == seed)
    ]

    if not existing.empty:
        print(f"Skipping already run configuration: "
              f"num_top_features={num_top_features}, hid={hid}, layers={layers}, lr={lr}, epochs={epochs}, k={k}, p={p}, r={r}")
        return True
    return False