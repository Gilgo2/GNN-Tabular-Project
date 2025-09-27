import numpy as np
from sklearn import datasets
import torch
import scipy.sparse as sp
import warnings
import pickle as pkl
import sys, os
import networkx as nx
import torch


class UCI():

    def __init__(self, n_train, n_val, dataset_name, scale=True, seed=42):
        self.n_train = n_train
        self.n_val = n_val
        self.dataset_name = dataset_name
        self.seed = seed
        self.scale = scale
        

    def load(self):
        if self.dataset_name == 'iris':
            data = datasets.load_iris()
        elif self.dataset_name == 'wine':
            data = datasets.load_wine()
        elif self.dataset_name == 'breast_cancer':
            data = datasets.load_breast_cancer()
        elif self.dataset_name == 'digits':
            data = datasets.load_digits()
        elif self.dataset_name == '20news10':
            from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.feature_extraction.text import TfidfTransformer
            categories = ['alt.atheism',
                          'comp.sys.ibm.pc.hardware',
                          'misc.forsale',
                          'rec.autos',
                          'rec.sport.hockey',
                          'sci.crypt',
                          'sci.electronics',
                          'sci.med',
                          'sci.space',
                          'talk.politics.guns']
            data = fetch_20newsgroups(subset='all', categories=categories)
            vectorizer = CountVectorizer(stop_words='english', min_df=0.05)
            X_counts = vectorizer.fit_transform(data.data).toarray()
            transformer = TfidfTransformer(smooth_idf=False)
            features = transformer.fit_transform(X_counts).todense()
            data['feature_names'] = np.arange(0, features.shape[1])
        else:
            raise AttributeError('dataset not available')

        if self.dataset_name != 'fma':
            from sklearn.preprocessing import scale
            if self.dataset_name != '20news10':
                if self.scale:
                    features = scale(data.data)
                else:
                    features = data.data
            y = data.target
        else:
            features = data['X']
            y = data['y']
        # ys = LabelBinarizer().fit_transform(y)
        # if ys.shape[1] == 1:
        #     ys = np.hstack([ys, 1 - ys])
        n = features.shape[0]
        idx = np.arange(n)

        from sklearn.model_selection import train_test_split
        train, test, y_train, y_test = train_test_split(np.arange(n), y, random_state=self.seed,
                                                        train_size=self.n_train + self.n_val,
                                                        test_size=n - self.n_train - self.n_val,
                                                        stratify=y)
        train, val, y_train, y_val = train_test_split(train, y_train, random_state=self.seed,
                                                      train_size=self.n_train, test_size=self.n_val,
                                                      stratify=y_train)
        
        train_mask = np.zeros([n, ], dtype=bool)
        train_mask[train] = True
        val_mask = np.zeros([n, ], dtype=bool)
        val_mask[val] = True
        test_mask = np.zeros([n, ], dtype=bool)
        test_mask[test] = True

        return features, data['feature_names'], y, train_mask, val_mask, test_mask
    

    warnings.simplefilter("ignore")

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_citation_network(dataset_str, sparse=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not sparse:
        adj = np.array(adj.todense(),dtype='float32')
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train = sample_mask(idx_train, labels.shape[0])
    val = sample_mask(idx_val, labels.shape[0])
    test = sample_mask(idx_test, labels.shape[0])

    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels)
    n = features.shape[0]
    train_mask = np.zeros([n, ], dtype=bool)
    train_mask[train] = True
    val_mask = np.zeros([n, ], dtype=bool)
    val_mask[val] = True
    test_mask = np.zeros([n, ], dtype=bool)
    test_mask[test] = True

    print("Training samples: ", train_mask.sum())
    print("Validation samples: ", val_mask.sum())
    print("Test samples: ", test_mask.sum())

    nfeats = features.shape[1]
    for i in range(labels.shape[0]):
        sum_ = torch.sum(labels[i])
        if sum_ != 1:
            labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
    labels = (labels == 1).nonzero()[:, 1]
    
    feature_names = np.arange(0, features.shape[1])
    return features, feature_names, labels, train_mask, val_mask, test_mask

def load_graph_data(dataset, sparse=0):
    return load_citation_network(dataset, sparse)

def get_data(dataset_name, seed=42):

    if dataset_name == 'wine':
        data_config = UCI(seed=seed, dataset_name=dataset_name, n_train=10, n_val=20, scale=True)
    elif dataset_name == 'breast_cancer':
        data_config = UCI(seed=seed, dataset_name=dataset_name, n_train=10, n_val=20, scale=True)
    elif dataset_name == 'digits':
        data_config = UCI(seed=seed, dataset_name=dataset_name, n_train=50, n_val=100, scale=False)
    elif dataset_name == '20newstrain':
        data_config = UCI(seed=seed, dataset_name=dataset_name, n_train=200, n_val=400, scale=False)
    elif dataset_name == '20news10':
        data_config = UCI(seed=seed, dataset_name=dataset_name, n_train=100, n_val=200, scale=False)
    elif dataset_name == 'cora':
        return load_graph_data('cora', sparse=0)
    elif dataset_name == 'citeseer':
        return load_graph_data('citeseer', sparse=0)
    return data_config.load()
