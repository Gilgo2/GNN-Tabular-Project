import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATASETS = ['cora', 'citeseer', 'wine', 'breast_cancer', 'digits', '20news10']


def parse_setting(row, num_features):
    """Return a human-readable setting name for a row."""
    if pd.notna(row.get('r')):
        return f"regular-{row['r']}"
    if pd.notna(row.get('p')):
        return f"random-{row['p']}"
    if pd.notna(row.get('num_top_features')) and row['num_top_features'] > 0:
        # handle KNN features
        features = (
            'all'
            if row['num_top_features'] >= num_features
            else str(int(row['num_top_features']))
        )
        return f"knn-k={int(row['k'])}-features={features}"
    return 'graphless'

def print_latex_table(df):
    for col in [c for c in df.columns if c.endswith('_mean')]:
        std_col = col.replace('_mean', '_std')
        df[col] = (
            (df[col] * 100).round(1).astype(str)
            + ' $\\pm$ '
            + (df[std_col] * 100).round(1).astype(str)
        )
        df.drop(columns=std_col, inplace=True)

    print(df.to_latex(index=False, float_format="%.2f"))

def empirical_results():
    # Collect mean/std per setting per dataset
    result_df = pd.DataFrame()

    for dataset in DATASETS:
        # Load results for each dataset
        perf_df = pd.read_csv(
            f"empirical/runs/{dataset}/graphconv_Adam_sum_hyperparam_search_results.csv"
        )
        top_features_df = pd.read_csv(
            f"empirical/runs/{dataset}/feature_importances.csv"
        )
        num_features = top_features_df.shape[0]

        # Compute setting names
        perf_df['setting'] = perf_df.apply(
            lambda x: parse_setting(x, num_features), axis=1
        )

        # Aggregate across seeds
        seed_df = (
            perf_df.groupby(['setting', 'seed'], as_index=False)
            .agg({"cost": "max"})
        )
        grouped = (
            seed_df.groupby('setting', as_index=False)
            .agg(mean=('cost', 'mean'), std=('cost', 'std'))
            .sort_values(by='mean')
        )

        # Build combined DataFrame
        if 'setting' not in result_df:
            result_df['setting'] = grouped['setting']
        result_df[f'{dataset}_mean'] = grouped['mean']
        result_df[f'{dataset}_std'] = grouped['std']

    print_latex_table(result_df.copy())
    

    # Plot KNN feature trend for citeseer
    knn_df = result_df
    knn_df = knn_df[knn_df['setting'].str.contains('knn|graphless')]
    knn_df['k'] = knn_df['setting'].apply(
        lambda x: int(x.split('-')[1].split('=')[1])
        if 'features' in x
        else 0
    )
    knn_df['num_features'] = knn_df['setting'].apply(
        lambda x: int(
            x.split('-')[2].split('=')[1].replace('all', '300')
        )
        if 'features' in x
        else 0
    )

    dataset_col = "citeseer_mean"
    plt.figure(figsize=(4, 3))

    k_perf = (
        knn_df.groupby('num_features')[dataset_col]
        .agg(['mean', 'std'])
        .reset_index()
    )

    plt.plot(
        k_perf['num_features'],
        k_perf['mean'],
        'o-',
        linewidth=2,
        markersize=8,
        color='blue',
    )
    plt.fill_between(
        k_perf['num_features'],
        k_perf['mean'] - k_perf['std'],
        k_perf['mean'] + k_perf['std'],
        alpha=0.2,
        color='blue',
    )
    plt.xlabel('Num Features', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Citeseer', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.14, right=0.98)

    plt.savefig('plots/num_features_trend.png', dpi=300)

empirical_results()