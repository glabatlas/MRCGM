import os
import pickle
from collections import Counter 
from functools import partial
import torch as t
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph

from subgraphx.explainer_core import (
    MCTS, MCTSConfig, RewardConfig, find_closest_node_result,
    gnn_score, GnnNets_NC2value_func,
    layered_sampling, z_score_normalization, calculate_new_weights
)
from utils import get_node_idx, get_node_name, get_name   # 确保 get_node_name 已导入


def get_khop_subgraph(node_idx, data, num_hops=1):
    """提取 1-hop 子图，并计算新边权重"""
    node_idx_t = t.tensor([node_idx], dtype=t.long)
    subset, edge_index_sub, _, edge_mask = k_hop_subgraph(
        node_idx=node_idx_t,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
        flow='target_to_source'
    )
    k_data = Data(
        x=data.x[subset],
        edge_index=edge_index_sub,
        edge_attr=data.edge_attr[edge_mask],
        y=data.y[subset]
    )

    k_data.ori_edge_attr = k_data.edge_attr.clone()
    k_data = z_score_normalization(k_data)
    new_weights = calculate_new_weights(k_data)
    k_data.edge_attr = new_weights

    subset_list = subset.tolist()
    original_to_sub = {orig: sub for sub, orig in enumerate(subset_list)}
    sub_to_original = {sub: orig for sub, orig in enumerate(subset_list)}
    return k_data, original_to_sub, sub_to_original

def run_subgraphx_for_genes(gene_list, model, dataset, iteration, top_k, subgraph_max_nodes, configs):
    device = configs['device']
    model.eval()
    model.to(device)
    data = dataset[0]

    out_dir = os.path.join(configs['out_dir'], configs['dataset_name'], f'iter_{iteration}', 'subgraphx')
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'subgraphx_results.pkl')
    txt_path = os.path.join(out_dir, 'subgraphx_results.txt')


    results = {}
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded existing results for {len(results)} genes from {pkl_path}")

    processed_genes = set(results.keys())
    genes_to_process = [g for g in gene_list if g not in processed_genes]

    if not genes_to_process:
        print("All genes already processed. Loading cached results.")
        return results

    print(f"Processing {len(genes_to_process)} genes out of {len(gene_list)} total.")

    node_idx_list = get_node_idx(genes_to_process)

    for node_idx in node_idx_list:
        gene_name = get_name(node_idx)
        print(f"Processing gene: {gene_name} (index {node_idx})")


        k_data, original_to_sub, sub_to_original = get_khop_subgraph(node_idx, data)

        all_sampled_indices = []
        num_epochs = max(1, k_data.x.shape[0] // 2)

        for _ in range(num_epochs):
            # 2. 分层采样
            resampled_data, new_ori2sub, new_sub2ori = layered_sampling(
                k_data, node_idx, original_to_sub, sub_to_original,
                sample_size=20)

            # 3. 转换为 NetworkX 图
            graph = to_networkx(resampled_data, to_undirected=True)
            node_labels = {k: int(v[1].item()) for k, v in enumerate(resampled_data.y)}
            nx.set_node_attributes(graph, node_labels, 'label')

            new_sub_node = next((k for k, v in new_sub2ori.items() if v == node_idx), None)
            new_sub_node = t.tensor([new_sub_node], dtype=t.long)

            value_func = GnnNets_NC2value_func(model, node_idx=new_sub_node, target_class=1)
            score_func = partial(gnn_score, value_func=value_func,
                                 subgraph_building_method=RewardConfig.subgraph_building_method)

            mcts = MCTS(
                node_idx=new_sub_node,
                ori_graph=graph,
                X=resampled_data.x,
                edge_index=resampled_data.edge_index,
                edge_attr=resampled_data.edge_attr,
                num_hops=2,
                n_rollout=MCTSConfig.rollout,
                min_atoms=MCTSConfig.min_atoms,
                c_puct=MCTSConfig.c_puct,
                expand_atoms=MCTSConfig.expand_atoms,
                score_func=score_func
            )

            gnn_results = mcts.mcts(verbose=False)
            best_node = find_closest_node_result(gnn_results, subgraph_max_nodes)

            sampled_coalition_idx = [mcts.reverse_mapping[idx] for idx in best_node.coalition]
            original_indices = [new_sub2ori[idx] for idx in sampled_coalition_idx]
            all_sampled_indices.extend(original_indices)


        counter = Counter(all_sampled_indices)
        top20_indices = [item[0] for item in counter.most_common(20)]
        results[gene_name] = top20_indices


        top20_genes = get_node_name(top20_indices)
        print(f"  Finished {gene_name}, collected {len(all_sampled_indices)} nodes, selected top 20: {top20_genes}")

        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)

        with open(txt_path, 'w') as f:
            for gene, idx_list in results.items():
                gene_list = get_node_name(idx_list)
                f.write(f"{gene}\t{repr(gene_list)}\n")
        print(f"  Checkpoint saved for {gene_name}.")

    print(f"SubgraphX completed for all genes. Results saved to {out_dir}")
    return results

