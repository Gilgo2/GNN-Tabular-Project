import os 
from tqdm import tqdm
import time
from utils import save_model, set_data_masks, did_run_already, save_best_tabular_model_feature_importance, df_to_graph_data
from train import train_gnn
from data_utils import get_data
import pandas as pd
import itertools
import argparse

def run_experiment(result_df, param_names, param_values, df, train_mask, val_mask, test_mask,
                   dataset_name, seed):
    """Run a GNN experiment over a param grid."""

    for param_tuple in tqdm(list(itertools.product(*param_values))):
        params = dict(zip(param_names, param_tuple))
        epochs = params['epochs']
        num_top_features = params.get('num_top_features')
        hid, layers, lr = params['hid'], params['layers'], params['lr']
        k, p, r = params.get('k'), params.get('p'), params.get('r')

        if did_run_already(result_df, num_top_features, hid, layers, lr, epochs, k, p, r, seed):
            continue

        data = df_to_graph_data(
            df, num_top_features=num_top_features,
            dataset_name=dataset_name, seed=seed, k=k, p=p, r=r
        )
        data = set_data_masks(data, train_mask, val_mask, test_mask)

        start_time = time.time()
        model, cost, epoch_converged = train_gnn(data, epochs, layers, hid)

        save_model(
            model, p, k, r, num_top_features,
            output_name=f'empirical/runs/{dataset_name}/models/graphconv_hid={hid}_layers={layers}_lr={lr}_epochs={epochs}_seed={seed}'
        )

        elapsed_time = time.time() - start_time
        temp_df = pd.DataFrame({
            'num_top_features': [num_top_features], 'hid': [hid], 'layers': [layers],
            'cost': [cost], 'epoch_converged': [epoch_converged], "total_time": [elapsed_time],
            'epoch': epochs, 'lr': lr, 'k': k, 'p': p, 'r': r,
            'train_count': train_mask.sum(), 'seed': seed
        })

        result_df = pd.concat([result_df, temp_df], ignore_index=True)
        result_df.to_csv(f'empirical/runs/{dataset_name}/graphconv_hyperparam_search_results.csv', index=False)

    return result_df

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='breast_cancer', help='Dataset name', choices=['breast_cancer', 'wine', 'digits', '20news10', 'cora', 'citeseer'])
    args = argparser.parse_args()
    dataset_name = args.dataset
    save_importance = True

    for seed in [42]:
        os.makedirs(f'empirical/runs/{dataset_name}/models', exist_ok=True)

        features, feature_names, y, train_mask, val_mask, test_mask = get_data(dataset_name, seed)
        df = pd.concat([pd.DataFrame(features, columns=feature_names), pd.DataFrame(y, columns=['target'])], axis=1)

        if save_importance:
            save_best_tabular_model_feature_importance(df, train_mask, val_mask, test_mask, dataset_name)
            save_importance = False

        result_file = f'empirical/runs/{dataset_name}/graphconv_hyperparam_search_results.csv'
        if os.path.exists(result_file):
            result_df = pd.read_csv(result_file)
        else:
            result_df = pd.DataFrame(columns=['num_top_features', 'hid', 'layers', 'cost', 'epoch_converged',
                                                'total_time', 'epoch', 'lr'])

        # define param grids
        knn_param_grid = {
            'epochs': [1000],
            'num_top_features': [0, 1, 3, 5, 10, 100, 300],
            'hid': [64], 'layers': [2, 3], 'lr': [0.01, 0.001],
            'k': [1, 5, 10, 15, 20, 25, 30, 35, 40]
        }
        if df.shape[1] < 100:
            knn_param_grid['num_top_features'] = knn_param_grid['num_top_features'][:-1]

        random_param_grid = {
            'epochs': [1000], 'hid': [64], 'layers': [2, 3],
            'lr': [0.001, 0.01], 'p': [0.003, 0.01]
        }
        regular_param_grid = {
            'epochs': [1000], 'hid': [64], 'layers': [2, 3],
            'lr': [0.001, 0.01], 'r': [2, 10, 50]
        }

        for param_grid in [knn_param_grid, random_param_grid, regular_param_grid]:
            result_df = run_experiment(
                result_df,
                list(param_grid.keys()), list(param_grid.values()),
                df, train_mask, val_mask, test_mask,
                dataset_name, seed
            )

        result_df.to_csv(result_file, index=False)


if __name__ == "__main__":
    main()