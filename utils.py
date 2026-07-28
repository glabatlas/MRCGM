import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, roc_auc_score

def minmax(input, axis=1):

    scaler = MinMaxScaler()
    if axis == 1:
        output = scaler.fit_transform(input)
    elif axis == 0:
        output = scaler.fit_transform(input.T).T
    elif axis == -1:
        output = (input - np.min(input)) / (np.max(input) - np.min(input))

    return output


def sum_norm(input, axis=1):

    axis_sum = input.sum(axis=1 - axis, keepdims=True)
    return input / axis_sum


def get_all_nodes(pan=False):
    dir = "data/Gene-Name.txt" if not pan else f"data/{pan}/Gene-Name.txt"
    with open(dir) as f:
        txt = f.readlines()
    gene_list = [line.strip() for line in txt[1:]]  # skip the first line
    gene_set = set(gene_list)

    return gene_list, gene_set


def get_labeled_nodes(cancer, labels=[-1, 1], return_labels=False, patient=None):

    HIC_NAMES = {"Breast_Cancer": "MCF7", "Leukemia": "K562"}
    file_name = f"data/{cancer}_Matrix/{HIC_NAMES[cancer]}-Label.txt" if not patient else f"data/{cancer}_Matrix/{patient}/{patient}-Label.txt"
    data = pd.read_csv(file_name, sep='\t')
    data = data[data['label'] != 0]
    data = data[data['label'].isin(labels)]
    node_list = data['gene_name'].to_list()
    print(len(node_list))

    if not return_labels:
        return node_list
    label_list = data['label'].to_list()

    return node_list, label_list


def get_gene_list(rename=False):
    gene_list = pd.read_csv("data/Gene-Name.txt")
    return gene_list.rename(columns={'gene_name': 'Gene Name'}) if rename else gene_list


def get_node_idx(node_list):

    dir = "data/Gene-Name.txt"
    gene_list = pd.read_csv(dir)
    gene_list.set_index('gene_name', inplace=True)
    gene_index = np.arange(gene_list.shape[0])
    gene_list['gene_index'] = gene_index
    node_idx_list = []
    for node in node_list:
        if node not in gene_list.index:
            print(f"{node} is not in gene list!")
            continue
        node_idx = gene_list.loc[node, 'gene_index']
        node_idx_list.append(node_idx)

    return node_idx_list


def get_node_name(node_idx_list):

    gene_list = pd.read_csv("data/Gene-Name.txt")
    assert all(x >= 0 and x < gene_list.shape[0] for x in node_idx_list), "Elements are out of bounds."

    node_name_list = gene_list.iloc[node_idx_list, 0].to_list()

    return node_name_list


def get_name(node_idx):

    gene_list = pd.read_csv("data/Gene-Name.txt")

    if np.isscalar(node_idx):
        node_idx = [int(node_idx)]  # 显式转换为Python原生类型

        # 后续逻辑保持不变
    assert all(x >= 0 and x < gene_list.shape[0] for x in node_idx), "Elements are out of bounds."
    node_name_list = gene_list.iloc[node_idx, 0].to_list()

    return node_name_list[0] if len(node_name_list) == 1 else node_name_list



def calculate_metrics(y_true, y_pred, y_score):

    num_correct = np.equal(y_true, y_pred).sum()
    acc = num_correct / y_true.shape[0]
    cf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auprc = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score)
    return acc, cf_matrix, auprc, f1, auroc