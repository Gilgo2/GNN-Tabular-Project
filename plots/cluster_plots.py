import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def heatmap(df_heat, out_folder, cluster_distance_multiplier=6):
    df_heat = df_heat[df_heat['cluster_distance_multiplier'] == cluster_distance_multiplier].drop('cluster_distance_multiplier', axis=1)
    sns.set_theme(context='talk', style='whitegrid')

    col = df_heat['Full']
    df_heat.drop('Full', axis=1, inplace=True)
    df_heat['Full'] = col
    
    plt.figure(figsize=(12, 6))

    sns.heatmap(
        df_heat,
        annot=True,
        fmt='.3g',
        cmap='YlGnBu',
        cbar_kws={'label': 'MAE'}
    )
    plt.xlabel('Cluster Size')
    plt.ylabel('kNN Graph K')

    plt.tight_layout()
    plt.savefig(f'{out_folder}/heatmaps.png', dpi=300)

def latex(grouped_df):


    # Build displayed columns with mean ± std
    metric_cols = [
        'final_classification_test_accuracy',
        'final_regression_test_mae',
        'final_classification_test_5_cluster_accuracy',
        'final_regression_5_cluster',
        'final_classification_test_10_cluster_accuracy',
        'final_regression_10_cluster',
        'final_classification_test_accuracy_std',
        'final_regression_test_mae_std',
        'final_classification_test_5_cluster_accuracy_std',
        'final_regression_5_cluster_std',
        'final_classification_test_10_cluster_accuracy_std',
        'final_regression_10_cluster_std'
    ]

    display_cols = []
    for col in metric_cols:
        if col.endswith('_std'):
            continue
        std_col = f"{col}_std"
        is_acc = 'accuracy' in col
        mean_vals = grouped_df[col]
        std_vals = grouped_df[std_col]
        if is_acc:
            mean_fmt = (mean_vals*100).round(0).astype('Int64').astype(str) + '%'
            std_fmt = (std_vals*100).round(0).astype('Int64').astype(str) + '%'
        else:
            mean_fmt = mean_vals.round(3).map(lambda x: f"{x:.3f}")
            std_fmt = std_vals.round(3).map(lambda x: f"{x:.3f}")
        grouped_df[col] = mean_fmt + ' $\\pm$ ' + std_fmt
        display_cols.append(col)

    # Bold the best per metric (min for regression/MAE, max for accuracy)
    for col in display_cols:
        is_reg = 'regression' in col
        if is_reg:
            best_mask = grouped_df[col] == grouped_df[col].min()
        else:
            best_mask = grouped_df[col] == grouped_df[col].max()
        grouped_df.loc[best_mask, col] = grouped_df.loc[best_mask, col].apply(lambda s: f"\\textbf{{{s}}}")

    # Final column order
    grouped_df = grouped_df[['ml_model', 'cluster_distance_multiplier',
                             'final_classification_test_5_cluster_accuracy', 'final_regression_5_cluster',
                             'final_classification_test_10_cluster_accuracy','final_regression_10_cluster',
                             'final_classification_test_accuracy', 'final_regression_test_mae']]

    print(grouped_df.to_latex(index=False, escape=False))

def line_plot(grouped_df, out_folder):
    mask = ~grouped_df['ml_model'].str.contains('|'.join(['k=10','rand=0.1','reg=4', 'reg=10']))
    grouped_df = grouped_df[mask]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)

    # Left: Full MAE
    sns.lineplot(
        data=grouped_df,
        x='cluster_distance_multiplier',
        y='final_regression_test_mae',
        hue='ml_model',
        marker='o',
        linewidth=2.0,
        markersize=7,
        palette='Set2',
        ax=axes[0]
    )
    axes[0].set_title('Full MAE', fontsize=12)
    axes[0].set_xlabel('Cluster Distance Multiplier')
    axes[0].set_ylabel('MAE')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title='Model', loc='lower right', fontsize=9)

    # Right: 5-cluster MAE
    sns.lineplot(
        data=grouped_df,
        x='cluster_distance_multiplier',
        y='final_regression_5_cluster',
        hue='ml_model',
        marker='s',
        linewidth=2.0,
        markersize=7,
        palette='Set2',
        ax=axes[1]
    )
    axes[1].set_title('5-Cluster MAE', fontsize=12)
    axes[1].set_xlabel('Cluster Distance Multiplier')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend().set_visible(False)

    plt.savefig(f'{out_folder}/lineplots.png', dpi=300)


def synthetic_results(in_path='cluster/runs/results.json', out_folder='plots'):
    pdf = pd.read_json(in_path)
    

    agg_cols = ['model_type', 'knn', 'p', 'r', 'cluster_distance_multiplier', 'seed']
    mae_cols = ['final_regression_test_mae', 'final_regression_5_cluster', 'final_regression_10_cluster','final_regression_15_cluster']
    pdf[agg_cols] = pdf[agg_cols].fillna('')
    
    agg_cols_no_seed = agg_cols[:-1]
    
    print(pdf[agg_cols + mae_cols].info())
    grouped_min =pdf[agg_cols + mae_cols].groupby(agg_cols, as_index=False).min()
    grouped_mae = (
        grouped_min.groupby(agg_cols_no_seed, as_index=False)
                   .mean()[agg_cols_no_seed + mae_cols]
    )
    pdf['ml_model'] = pdf['ml_model'].fillna('')

    conv_df = grouped_mae[
        (grouped_mae['model_type'] == 'gnn') & (grouped_mae['p'] == '') & (grouped_mae['r'] == '')
    ][['knn', 'cluster_distance_multiplier'] + mae_cols].copy()

    conv_df = conv_df.rename(columns={
        'final_regression_5_cluster': '5',
        'final_regression_10_cluster': '10',
        'final_regression_15_cluster': '15',
        'final_regression_test_mae': 'Full'
    })

    df_heat = conv_df.set_index('knn')
    heatmap(df_heat, out_folder)

    pdf_tab = pdf.copy()
    pdf_tab.loc[pdf_tab['ml_model'] == 'rf', 'ml_model'] = 'RF'
    pdf_tab.loc[pdf_tab['ml_model'] == 'mlp', 'ml_model'] = 'MLP'
    pdf_tab.loc[pdf_tab['ml_model'] == '', 'ml_model'] = 'Conv'

    table_df = pdf_tab.groupby(['ml_model', 'seed', 'knn', 'p', 'r', 'cluster_distance_multiplier'], as_index=False)\
               .agg({'final_classification_test_accuracy':'max','final_regression_test_mae':'min',
                   'final_classification_test_5_cluster_accuracy':'max', 'final_regression_5_cluster':'min',
                   'final_classification_test_10_cluster_accuracy':'max', 'final_regression_10_cluster':'min',
                   'final_classification_test_15_cluster_accuracy':'max', 'final_regression_15_cluster':'min'})

    grouped_df = table_df.groupby(['ml_model', 'knn', 'p', 'r', 'cluster_distance_multiplier'], as_index=False)\
    .agg(final_classification_test_accuracy=("final_classification_test_accuracy","mean"),
          final_regression_test_mae=("final_regression_test_mae","mean"),
          final_classification_test_5_cluster_accuracy=("final_classification_test_5_cluster_accuracy","mean"),
            final_regression_5_cluster=("final_regression_5_cluster","mean"),
          final_classification_test_10_cluster_accuracy=("final_classification_test_10_cluster_accuracy","mean"),
            final_regression_10_cluster=("final_regression_10_cluster","mean"),
          final_classification_test_15_cluster_accuracy=("final_classification_test_15_cluster_accuracy","mean"),
            final_regression_15_cluster=("final_regression_15_cluster","mean"),
          final_classification_test_accuracy_std=("final_classification_test_accuracy","std"),
            final_regression_test_mae_std=("final_regression_test_mae","std"),
          final_classification_test_5_cluster_accuracy_std=("final_classification_test_5_cluster_accuracy","std"),
            final_regression_5_cluster_std=("final_regression_5_cluster","std"),
          final_classification_test_10_cluster_accuracy_std=("final_classification_test_10_cluster_accuracy","std"),
            final_regression_10_cluster_std=("final_regression_10_cluster","std"),
          final_classification_test_15_cluster_accuracy_std=("final_classification_test_15_cluster_accuracy","std"),
            final_regression_15_cluster_std=("final_regression_15_cluster","std"))

    # Remove 15-cluster columns per original logic
    grouped_df = grouped_df[[c for c in grouped_df.columns if '_15_' not in c]]

    def setting_label(ml_model, knn, p, r):
        if ml_model != 'Conv':
            return ''
        if r != '':
            return f'reg={int(r)}'
        if p != '':
            return f'rand={p}'
        if knn != 0:
            return f'k={int(knn)}'
        return 'graphless'

    grouped_df['setting'] = grouped_df.apply(lambda x: setting_label(x['ml_model'], x['knn'], x['p'], x['r']), axis=1)
    grouped_df['ml_model'] = grouped_df.apply(
        lambda x: x['ml_model'] + "$_{" + x['setting'] + "}$" if x['setting'] != '' else x['ml_model'], axis=1
    )
    latex(grouped_df.copy())
    line_plot(grouped_df.copy(), out_folder)

    

if __name__ == '__main__': 
    
    synthetic_results()
