import copy
import torch
import numpy as np
import networkx as nx
from collections import Counter
from functools import partial
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import k_hop_subgraph   # 也可不导入，但保留


class MCTSConfig:
    rollout = 10
    high2low = False
    c_puct = 5.0
    min_atoms = 3
    expand_atoms = 10

class RewardConfig:
    reward_method = 'gnn_score'
    subgraph_building_method = 'zero_filling'

# ---------------------------- MCTSNode  ----------------------------
class MCTSNode:
    def __init__(self, coalition, data=None, ori_graph=None, c_puct=5.0):
        self.coalition = coalition
        self.children = []
        self.N = 0
        self.W = 0.0
        self.P = 0.0
        self.data = data
        self.ori_graph = ori_graph
        self.c_puct = c_puct

    def Q(self):

        return self.W / (self.N + 1e-8)

    def U(self, sum_count):
        """U = c_puct * P * sqrt(sum_count) / (1 + N)"""
        return self.c_puct * self.P * np.sqrt(sum_count) / (1 + self.N)


def find_closest_node_result(results, max_nodes):
    """从 MCTS 结果中选出节点数不超过 max_nodes 且价值最高的子图"""
    results = sorted(results, key=lambda x: x.P, reverse=True)
    results = sorted(results, key=lambda x: len(x.coalition))
    result_node = results[0]
    for r in results:
        if len(r.coalition) <= max_nodes and r.P > result_node.P:
            result_node = r
    return result_node


def graph_build_zero_filling(X, edge_index, node_mask):
    """将未选中节点的特征置零"""
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index

def gnn_score(coalition, data, value_func, subgraph_building_method='zero_filling'):
    """计算选定节点子图对目标节点的预测得分"""
    num_nodes = data.num_nodes
    mask = torch.zeros(num_nodes, dtype=torch.float32)
    mask[coalition] = 1.0
    ret_x, _ = graph_build_zero_filling(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    score = value_func(mask_data)
    return score.item()

def GnnNets_NC2value_func(model, node_idx, target_class):
    def value_func(data):
        with torch.no_grad():
            device = data.x.device
            if isinstance(node_idx, int):
                node_idx_tensor = torch.tensor([node_idx], device=device)
            else:
                node_idx_tensor = node_idx.to(device)
            logits = model([data])   # 注意这里改为列表
            prob = torch.sigmoid(logits) if logits.size(-1) == 1 else torch.softmax(logits, dim=-1)
            if prob.dim() == 1:
                selected = prob[node_idx_tensor]
            else:
                selected = prob[node_idx_tensor, target_class] if prob.size(-1) > 1 else prob[node_idx_tensor]
            return selected if target_class == 1 else 1.0 - selected
    return value_func


def k_hop_subgraph_with_default_whole_graph(node_idx, num_hops,
    edge_index, edge_attr, relabel_nodes=False, num_nodes=None, flow='source_to_target'):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index[0], edge_index[1]
    else:
        col, row = edge_index[1], edge_index[0]

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if int(node_idx) == -1:   # 全图
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while True:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]
    edge_index = edge_index[:, edge_mask]
    filtered_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx_mapping = row.new_full((num_nodes,), -1)
        node_idx_mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx_mapping[edge_index]

    return subset, edge_index, filtered_edge_attr, inv, edge_mask

class MCTS:
    def __init__(self, node_idx, X, edge_index, edge_attr,
                     ori_graph, n_rollout, min_atoms, c_puct,
                     expand_atoms, score_func, num_hops=1):
        self.X = X
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.num_hops = num_hops
        self.ori_graph = ori_graph
        self.ori_node_idx = node_idx
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.score_func = score_func


        subset, edge_index_sub, edge_attr_sub, inv, edge_mask = k_hop_subgraph_with_default_whole_graph(
                node_idx, self.num_hops, self.edge_index, self.edge_attr,
                relabel_nodes=True, num_nodes=self.X.size(0))
        x_sub = self.X[subset]

        self.data = Data(x=x_sub, edge_index=edge_index_sub, edge_attr=edge_attr_sub)
        self.graph = self.ori_graph.subgraph(subset.tolist())
        self.mapping = {int(v): k for k, v in enumerate(subset)}
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.graph = nx.relabel_nodes(self.graph, self.mapping)
        self.node_idx = torch.where(subset == self.ori_node_idx)[0]
        self.num_nodes = self.graph.number_of_nodes()

        self.root_coalition = list(range(self.num_nodes))
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct)
        self.root = self.MCTSNodeClass(self.root_coalition)
        self.state_map = {str(sorted(self.root.coalition)): self.root}

    def set_score_func(self, score_func):
        self.score_func = score_func

    def mcts_rollout(self, tree_node):
        cur_graph_coalition = tree_node.coalition
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P

        if len(tree_node.children) == 0:

            node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
            node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=MCTSConfig.high2low)
            all_nodes = [x[0] for x in node_degree_list]
            if len(all_nodes) < self.expand_atoms:
                expand_nodes = all_nodes[:]
            else:
                expand_nodes = all_nodes[:self.expand_atoms]
            if self.node_idx in expand_nodes:
                expand_nodes.remove(self.node_idx)

            for each_node in expand_nodes:
                subgraph_coalition = [n for n in all_nodes if n != each_node]
                subgraphs = [self.graph.subgraph(c) for c in nx.connected_components(
                             self.graph.subgraph(subgraph_coalition))]
                main_sub = max(subgraphs, key=lambda g: g.number_of_nodes())
                new_graph_coalition = sorted(list(main_sub.nodes()))


                same_state = None
                for old_node in self.state_map.values():
                    if Counter(old_node.coalition) == Counter(new_graph_coalition):
                        same_state = old_node
                        break
                if same_state is None:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node
                else:
                    new_node = same_state

                if not any(Counter(child.coalition) == Counter(new_graph_coalition) for child in tree_node.children):
                    tree_node.children.append(new_node)


            for child in tree_node.children:
                if child.P == 0:
                    child.P = self.score_func(child.coalition, child.data)

        sum_count = sum(c.N for c in tree_node.children)
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True):
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root)
            if verbose:
                print(f"Rollout {rollout_idx}: {len(self.state_map)} states explored.")
        explanations = [node for _, node in self.state_map.items()]
        return sorted(explanations, key=lambda x: x.P, reverse=True)


def z_score_normalization(data):
    mean = data.x.mean(dim=0)
    std = data.x.std(dim=0)
    data.x = (data.x - mean) / std
    return data

def calculate_new_weights(data):
    edge_index = data.edge_index
    u, v = edge_index[0], edge_index[1]
    feats_u, feats_v = data.x[u], data.x[v]
    dot = (feats_u * feats_v).sum(dim=1)
    pcc = dot / (feats_u.shape[1] - 1)
    abs_pcc = torch.abs(pcc)
    original = data.edge_attr.squeeze()
    new_weights = 0.5 * abs_pcc + 0.5 * original
    return new_weights.unsqueeze(1)

def layered_sampling(subgraph, target_orig_node, original_to_sub, sub_to_original, sample_size=20):

    edge_index = subgraph.edge_index.numpy().T
    G = nx.Graph()
    if subgraph.edge_attr is not None:
        attrs = subgraph.edge_attr.numpy()
        for i, (u, v) in enumerate(edge_index):
            w = attrs[i].item() if attrs.ndim > 1 else attrs[i]
            G.add_edge(u, v, weight=w)
    else:
        G.add_edges_from(edge_index)
        nx.set_edge_attributes(G, 1.0, 'weight')

    target_sub_node = original_to_sub[target_orig_node]
    sampled_nodes = {target_sub_node}
    neighbors_edges = list(G.edges(target_sub_node, data='weight'))
    if neighbors_edges:
        neighbors, weights = [], []
        for u, v, w in neighbors_edges:
            nb = v if v != target_sub_node else u
            neighbors.append(nb)
            weights.append(w)
        need = sample_size - 1
        if len(neighbors) <= need:
            selected = neighbors
        else:
            probs = np.array(weights) / sum(weights)
            selected = np.random.choice(neighbors, size=need, replace=False, p=probs).tolist()
        sampled_nodes.update(selected)


    subgraph_nx = G.subgraph(sampled_nodes)
    if not nx.is_connected(subgraph_nx):
        comps = list(nx.connected_components(subgraph_nx))
        target_comp = next((c for c in comps if target_sub_node in c), set())
        other = sampled_nodes - target_comp
        if other:
            degrees = [G.degree(n, weight='weight') for n in other]
            if sum(degrees) > 0:
                probs = np.array(degrees) / sum(degrees)
                keep = np.random.choice(list(other), size=min(len(other), sample_size - len(target_comp)),
                                        replace=False, p=probs).tolist()
            else:
                keep = np.random.choice(list(other), size=min(len(other), sample_size - len(target_comp)),
                                        replace=False).tolist()
            target_comp.update(keep)
        sampled_nodes = target_comp

    new_sub_nodes = sorted(sampled_nodes)
    new_sub_to_original = {new: sub_to_original[old] for new, old in enumerate(new_sub_nodes)}
    new_original_to_sub = {orig: new for new, orig in new_sub_to_original.items()}

    node_mask = torch.tensor([(i in new_sub_nodes) for i in range(subgraph.num_nodes)], dtype=torch.bool)
    edge_mask = node_mask[subgraph.edge_index[0]] & node_mask[subgraph.edge_index[1]]
    resampled = Data(
        x=subgraph.x[node_mask],
        edge_index=subgraph.edge_index[:, edge_mask],
        edge_attr=subgraph.edge_attr[edge_mask] if subgraph.edge_attr is not None else None,
        y=subgraph.y[node_mask]
    )
    if hasattr(subgraph, 'ori_edge_attr'):
        resampled.edge_attr_original = subgraph.ori_edge_attr[edge_mask]

    node_id_map = {old: new for new, old in enumerate(new_sub_nodes)}
    mapped_edges = [[node_id_map[src], node_id_map[dst]] for src, dst in resampled.edge_index.t().tolist()]
    resampled.edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()
    return resampled, new_original_to_sub, new_sub_to_original