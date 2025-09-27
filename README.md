

This repository investigates when **Graph Neural Networks (GNNs)** outperform strong tabular baselines using controlled synthetic datasets, pinpointing the regimes and design choices where graph learning yields clear gains.
 It focuses on small-sample generalization, the role of local context, and the explores the effect of graph construction on downstream performance.

## Installation

Clone the repo and install dependencies:

The code uses PyTorch and PyTorch Geometric.
If you have a CUDA-capable GPU, install the CUDA version of PyTorch first:

```bash
pip install -r requirements
```

## Running the Project

### Cluster experiments
This task generates synthetic cluster like data points to generate a tabular dataset and examines GNN models vs traditional ML models.

Cluster experiments `run_experiment.py` accepts parameters and expects a wandb api key.
Main parameters used:
```bash
python cluster/run_experiment.py --model_type gnn --knn 4
python cluster/run_experiment.py --model_type ml --ml_model mlp
python cluster/run_experiment.py --model_type ml --ml_model rf
python cluster/run_experiment.py --model_type gnn --settings random --p 0.03
python cluster/run_experiment.py --model_type gnn --settings regular --r 10
```

#### Results:
- Generated data is saved in `cluster/data`
- Trained model is saved in `cluster/runs`
- Metrics are saved in `cluster/runs/results.json`

### GNNs vs Deepsets
This task compares **DeepSets** and several **Graph Neural Networks (GNNs)** (GCN, GraphSAGE, GIN, GAT) on synthetic datasets.  
It includes dataset generation, training, hyperparameter grid search, and testing

The main entry point is main.py:
```bash
python deepset/main.py
```

#### Results:
The testing function automatically saves evaluation outputs to the results/ directory.
⚠️ Note: the save path is currently hardcoded inside the code. If you want results in a different location, update the path in testing.py.

### Downstream tasks
This section compares our constructed GNN to ML models, baseline GNN models and GSL methods. 
Loading data code is taken from [LDS-GNN](https://github.com/lucfra/LDS-GNN/tree/master)


The entrypoint is run_experiment and it expects a dataset_name parameter out of [`breast_cancer`, `wine`, `digits`, `20news10`, `cora`, `citeseer`]
```bash
python empirical/run_experiment.py --dataset <dataset_name>
```
#### Results:
- Metrics and models are saved in `empirical/runs/{dataset_name}`


