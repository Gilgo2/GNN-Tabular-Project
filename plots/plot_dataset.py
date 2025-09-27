import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import pca
import numpy as np
from sklearn.decomposition import PCA

fig, axes = plt.subplots(2, 3, figsize=(4, 3))
axes = axes.flatten()
for i, multi in enumerate([0.5,1.0,6.0,12.0,25.0]):
    pdf = pd.read_csv(f'cluster/data/dataset_{multi}.csv')
    X = pdf.iloc[:, :-4].values
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    y_classification = pdf['y_classification']
    scatter = axes[i].scatter(X[:, 0], X[:, 1], 
                        c=y_classification, cmap='viridis', alpha=0.7, s=1)
    axes[i].set_title(f'{multi}', fontsize=8, pad=2)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.colorbar(scatter, cax=axes[5])
axes[5].set_title('Class', fontsize=8, pad=2)
plt.savefig('plots/pca_2d_subplots.png', bbox_inches='tight', dpi=300, pad_inches=0.1)


