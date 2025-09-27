import json
import argparse
from data_utils import generate_data, split_data
from train import train_ml_model, train_gnn_model
import wandb
import os
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Run ML/GNN experiment with wandb logging')
    parser.add_argument('--model_type', choices=['ml', 'gnn'], required=True,
                       help='Model type: ml or gnn')
    parser.add_argument('--ml_model', choices=['mlp', 'rf'], default='mlp',
                       help='ML model type (if model_type=ml): mlp or rf')
    parser.add_argument('--knn', type=int, default=20,
                       help='Graph edges')
    parser.add_argument('--cluster_distance_multiplier', type=float, default=6,
                       help='Overlap of clusters')
    parser.add_argument('--settings', default='knn',
                       help='Graph edges setting', choices=['knn','random','regular'])
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--iterations', type=int, default=5000,
                       help='Number of training iterations/epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--project', default='sgd_various_clusters_same_data',
                       help='WandB project name')
    parser.add_argument('--run_name', default=None,
                       help='WandB run name (optional)')
    parser.add_argument('--n_features', type=int, default=4,
                       help='Dimension of input')
    parser.add_argument('--r', type=int, default=5,
                       help='Degree of regular graphs')
    parser.add_argument('--p', type=float, default=0.03,
                       help='Probability for erdos renyi graphs')

    args = parser.parse_args()

    # Generate experiment name
    if args.run_name is None:
        if args.model_type == 'ml':
            model_name = args.ml_model
        else:
            model_name = 'conv'
        args.run_name = f"{args.model_type}_{model_name}_lr{args.learning_rate}_iter{args.iterations}_seed{args.seed}_nfeatures{args.n_features}"
        if args.model_type != 'ml':
            args.run_name += f'_k{args.knn}'

    cfg = {
            "model_type": args.model_type,
            "ml_model": args.ml_model if args.model_type == 'ml' else None,
            "gnn_model": 'conv' if args.model_type == 'gnn' else None,
            "hidden_dimensions": '128x128',
            "learning_rate": args.learning_rate,
            "iterations": args.iterations,
            "seed": args.seed,
            "n_communities": 10,
            "nodes_per_community": 30,
            "n_features": args.n_features,
            "setting": args.settings,
            'p': args.p if args.settings == 'random' else None,
            'r': args.r if args.settings == 'regular' else None,
            "knn": args.knn if args.settings == 'knn' else None,
            "cluster_distance_multiplier": args.cluster_distance_multiplier
        }
        
    # Initialize wandb
    wandb.init(
        project=args.project,
        name=args.run_name,
        config=cfg
    )

    print(f"Starting experiment: {args.run_name}")
    print(f"Model type: {args.model_type}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Iterations: {args.iterations}")
    print(f"Seed: {args.seed}")

    # Generate data
    if not os.path.exists(f'cluster/data/data_splits_{args.cluster_distance_multiplier}.json'):
       
        X, y_reg, y_class, _, _ = generate_data(
            n_communities=10, 
            nodes_per_community=30, 
            n_features=args.n_features, 
            random_state=args.seed,
            cluster_distance_multiplier=args.cluster_distance_multiplier
        )

        # Split data
        data_splits = split_data(X, y_reg, y_class, args.seed)
            # save data splits
        with open(f'cluster/data/data_splits_{args.cluster_distance_multiplier}.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in data_splits.items()}, f)
        print("Saved data splits to data_splits.json")
    else:
        print("Loading existing data splits from data_splits.json")
        df = pd.read_csv(f'cluster/data/dataset_{args.cluster_distance_multiplier}.csv')
        X = df.iloc[:, 0:-4].values
        y_reg = df['y_regression'].values
        y_class = df['y_classification'].values

        with open(f'cluster/data/data_splits_{args.cluster_distance_multiplier}.json') as f:
            data_splits_raw = json.load(f)
        data_splits = {k: np.array(v) for k, v in data_splits_raw.items()}

    # Train model
    if args.model_type == 'ml':
        results = train_ml_model(
            data_splits, 
            model_type=args.ml_model,
            learning_rate=args.learning_rate,
            max_iter=args.iterations,
            random_state=args.seed
        )
    else:  # gnn
        results = train_gnn_model(
            X, y_reg, y_class, data_splits, args.knn,
            n_features=args.n_features,
            n_classes=15,
            settings=args.settings,
            p=args.p,
            r=args.r,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            cluster_distance_multiplier=args.cluster_distance_multiplier
        )

    # Log final results
    for key, value in results['regression'].items():
        wandb.log({f"regression/{key}": value})
    for key, value in results['classification'].items():
        wandb.log({f"classification/{key}": value})
    
    wandb.finish()

    results.update(cfg)
    if os.path.exists('cluster/runs/results.json'):
        with open('cluster/runs/results.json') as f:
            results_f = json.load(f)
    else:
        results_f = []
    results_f.append(results)
    with open('cluster/runs/results.json', 'w') as f:
        json.dump(results_f, f, indent=4)


if __name__ == "__main__":
    main()
