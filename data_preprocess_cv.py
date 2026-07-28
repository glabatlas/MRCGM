import os
import pickle
import pandas as pd
import numpy as np
import torch as t
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data, InMemoryDataset
import config_load
from utils import minmax

EPS = 1e-8

def get_cell_line(data_dir):
    if "Breast_Cancer" in data_dir:
        return "/MCF7"
    elif "Leukemia" in data_dir:
        return "/K562"
    elif "AML" in data_dir:
        return "/" + os.path.basename(data_dir)
    else:
        raise ValueError(f"Unknown data_dir: {data_dir}")

def read_table_to_np(table_file, sep='\t', dtype=float, start_col=1):
    data = pd.read_csv(table_file, sep=sep)
    data = data.iloc[:, start_col:].to_numpy().astype(dtype)
    return data

def get_hic_mat(data_dir, reduce='svd', reduce_dim=5, resolution='10KB', type='ice', norm='log'):
    cell_line = get_cell_line(data_dir)
    hic_dir = data_dir + cell_line + "_Hi-C_Ice" if type == 'ice' else data_dir + cell_line + "_Hi-C"
    hic_data = read_table_to_np(hic_dir, sep='\t', dtype=float, start_col=1)
    if norm == 'log':
        hic_data = np.log10(hic_data + EPS)
    if reduce == 'svd':
        reducer = TruncatedSVD(n_components=reduce_dim, algorithm="arpack")
        hic_data = reducer.fit_transform(hic_data)
    return minmax(hic_data, axis=1)

def get_pathway_mat(data_dir, reduce='svd', reduce_dim=5):
    # 所有数据集共用同一个 Pathway 矩阵文件
    pathway_dir = "data/pathway/Pathway-Matrix.csv"
    if not os.path.exists(pathway_dir):
        raise FileNotFoundError(f"Pathway matrix not found at {pathway_dir}")
    pathway_data = read_table_to_np(pathway_dir, sep='\t', dtype=float, start_col=1)
    if reduce == 'svd':
        reducer = TruncatedSVD(n_components=reduce_dim, algorithm="arpack")
        pathway_data = reducer.fit_transform(pathway_data)
    return minmax(pathway_data, axis=1)

def get_ppi_mat(ppi_name='CPDB', ppi_path=None):
    if ppi_path is not None:
        print(f"Loading PPI matrix from {ppi_path} ......")
        data = pd.read_csv(ppi_path, sep='\t', index_col=0).to_numpy()
        return data

    ppi_dir = f"data/{ppi_name}/{ppi_name}_matrix.csv"
    print(f"Loading PPI matrix from {ppi_dir} ......")
    data = pd.read_csv(ppi_dir, sep="\t", index_col=0).to_numpy()
    return data

def get_label(data_dir):
    cell_line = get_cell_line(data_dir)
    label_dir = data_dir + cell_line + "-Label.txt"
    data = read_table_to_np(label_dir, dtype=int).transpose()[0]
    labels = np.zeros((len(data), 2), dtype=float)
    labeled_idx = []
    for i, val in enumerate(data):
        if val != 0:
            labeled_idx.append(i)
            if val == -1:
                labels[i][0] = 1
            else:
                labels[i][1] = 1
    return labels, labeled_idx

def get_node_feat(hic_feat=None, data_dir='data/Breast_Cancer_Matrix'):
    cell_line = get_cell_line(data_dir)
    feat = read_table_to_np(data_dir + cell_line + "-Normalized-Nodefeature-Matrix.csv", sep=',')
    feat = feat[:, :10]  # 前10个原始特征
    if hic_feat is not None:
        feat = np.concatenate((feat, hic_feat), axis=1)
    return feat, np.arange(feat.shape[0])

def construct_edge(mat):
    num_nodes = mat.shape[0]
    edges, val = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if mat[i, j] > 0:
                edges.append([i, j])
                val.append(mat[i, j])
    edge_index = np.transpose(edges)
    edge_attr = np.reshape(val, (-1, 1))
    return edge_index, edge_attr


def build_pyg_data(node_mat, node_lab, ppi_mat):
    x = t.tensor(node_mat, dtype=t.float)
    y = t.tensor(node_lab, dtype=t.long)
    edge_index, edge_attr = construct_edge(ppi_mat)
    edge_index = t.tensor(edge_index, dtype=t.long)
    edge_attr = t.tensor(edge_attr, dtype=t.float)

    if edge_attr.dim() == 3:
        edge_attr = edge_attr.squeeze(1)
    elif edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(1)

    pos = t.arange(node_mat.shape[0], dtype=t.long)
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    return data

class CancerDataset(InMemoryDataset):
    def __init__(self, data):
        super(CancerDataset, self).__init__('.', None, None)
        self.data = data
        self.data.num_classes = 2
        self.data, self.slices = self.collate([self.data])

    def get_idx_split(self, fold):
        train_mask = self.data.train_mask[:, fold]
        valid_mask = self.data.valid_mask[:, fold]
        test_mask = self.data.test_mask
        return {
            'train': t.where(train_mask)[0],
            'valid': t.where(valid_mask)[0],
            'test': t.where(test_mask)[0]
        }

def get_data(configs, ppi_path=None, load_existing=False):

    data_dir = configs['data_dir']
    dataset_name = configs['dataset_name']
    iteration = configs.get('iteration', 0)
    out_dir = os.path.join(configs['out_dir'], dataset_name, f'iter_{iteration}')
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'dataset.pkl')

    # 第一轮且存在原始数据集文件
    if iteration == 0 and load_existing:
        # 根据数据集类型构建默认pkl路径
        if "Breast_Cancer" in data_dir:
            default_pkl = f"data/Breast_Cancer_Matrix/{dataset_name}_CPDB_dataset_final.pkl"
        elif "Leukemia" in data_dir:
            default_pkl = f"data/Leukemia_Matrix/{dataset_name}_CPDB_dataset_final.pkl"
        elif "AML" in data_dir:
            default_pkl = f"data/AML_Matrix/{dataset_name}/{dataset_name}_CPDB_dataset_final.pkl"
        else:
            default_pkl = None

        if default_pkl and os.path.exists(default_pkl):
            with open(default_pkl, 'rb') as f:
                dataset = pickle.load(f)
            print(f"Loaded existing dataset from {default_pkl}")

            if dataset[0].x.shape[1] == 15 and configs.get('pathway', False):
                print("Detected 15-dim features, adding pathway features...")
                pathway_feat = get_pathway_mat(data_dir, reduce=configs['pathway_reduce'],
                                               reduce_dim=configs['pathway_reduce_dim'])
                print(f"Pathway feature shape: {pathway_feat.shape}")
                pathway_tensor = t.tensor(pathway_feat, dtype=t.float)

                new_x = t.cat([dataset.data.x, pathway_tensor], dim=1)
                dataset.data.x = new_x
                print(f"New feature dimension: {dataset.data.x.shape[1]}")

            return dataset

    # ========== 重新生成数据集 ==========
    # 1. Hi-C
    hic_feat = get_hic_mat(data_dir, reduce=configs['hic_reduce'],
                           reduce_dim=configs['hic_reduce_dim'],
                           resolution=configs['resolution'],
                           type=configs['hic_type'],
                           norm=configs['hic_norm']) if configs['hic'] else None

    # 2. 基础特征（前10个 + Hi-C）
    base_feat, pos = get_node_feat(hic_feat=hic_feat, data_dir=data_dir)

    # 3. Pathway 特征
    if configs.get('pathway', False):
        pathway_feat = get_pathway_mat(data_dir, reduce=configs['pathway_reduce'],
                                       reduce_dim=configs['pathway_reduce_dim'])
        print(f"Pathway feature shape during generation: {pathway_feat.shape}")
        base_feat = np.concatenate([base_feat, pathway_feat], axis=1)

    # 4. PPI 矩阵
    ppi_mat = get_ppi_mat(ppi_name=configs['ppi'], ppi_path=ppi_path)

    # 5. 标签
    node_lab, labeled_idx = get_label(data_dir)

    # 6. 划分 train/valid/test (10折)
    cv_folds = configs['cv_folds']
    random_seed = configs['random_seed']
    labeled_lab = [node_lab[i][1] for i in labeled_idx]
    train_valid_idx, test_idx, train_valid_lab, test_lab = train_test_split(
        labeled_idx, labeled_lab, test_size=0.25, stratify=labeled_lab, random_state=random_seed)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    train_idx_list, valid_idx_list = [], []
    for train_idx, valid_idx in skf.split(train_valid_idx, train_valid_lab):
        valid_idx_list.append([train_valid_idx[i] for i in valid_idx])
        train_idx_list.append([train_valid_idx[i] for i in train_idx])

    # 7. 构建 PyG Data
    data = build_pyg_data(base_feat, node_lab, ppi_mat)
    num_nodes = data.num_nodes
    train_mask = np.zeros((num_nodes, cv_folds), dtype=bool)
    valid_mask = np.zeros((num_nodes, cv_folds), dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for i in range(cv_folds):
        train_mask[train_idx_list[i], i] = True
        valid_mask[valid_idx_list[i], i] = True
    test_mask[test_idx] = True
    data.train_mask = t.tensor(train_mask, dtype=t.bool)
    data.valid_mask = t.tensor(valid_mask, dtype=t.bool)
    data.test_mask = t.tensor(test_mask, dtype=t.bool)
    data.unlabeled_mask = ~t.logical_or(data.train_mask[:, 0], t.logical_or(data.valid_mask[:, 0], data.test_mask))

    # 8. 保存数据集
    dataset = CancerDataset(data)
    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Generated dataset saved to {pkl_path}")

    return dataset


def update_dataset_ppi(dataset, new_ppi_matrix, save_path):
    data = dataset.data
    edge_index_np, edge_dim, edge_attr_np = construct_edge(new_ppi_matrix)
    data.edge_index = t.tensor(edge_index_np, dtype=t.long)
    data.edge_attr = t.tensor(edge_attr_np, dtype=t.float)
    data.edge_dim = edge_dim

    if hasattr(data, 'pos') and data.pos is not None:
        if data.pos.dim() == 2:
            data.pos = data.pos.view(-1)
    else:
        data.pos = t.arange(data.num_nodes, dtype=t.long)

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)