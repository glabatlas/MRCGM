import argparse
import os
import time
import json
import glob
import pickle
import numpy as np
import pandas as pd
import torch as t
from multiprocessing import Pool
from functools import partial

import config_load
from model import MRCGM
from JCMA.Attn import JCMA
from train import train_model, predict_all_scores_cv, load_best_model
from subgraphx.sub import run_subgraphx_for_genes
from utils import get_gene_list, get_node_idx, get_node_name
from data_preprocess_cv import get_data, CancerDataset,update_dataset_ppi

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

def update_ppi_matrix(ppi_path, gene_selected_dict, out_path):

    ppi_df = pd.read_csv(ppi_path, index_col=0, sep='\t')
    ppi_df.index = ppi_df.index.astype(str)
    ppi_df.columns = ppi_df.columns.astype(str)
    modified = ppi_df.copy()
    for gene, selected_list in gene_selected_dict.items():
        if gene not in modified.index:
            continue
        for target in set(selected_list):
            if target == gene or target not in modified.columns:
                continue
            old_w = modified.loc[gene, target]
            new_w = old_w + np.exp(old_w) / (1 + np.exp(old_w))
            modified.loc[gene, target] = new_w
            modified.loc[target, gene] = new_w
    modified.to_csv(out_path, sep='\t')
    print(f"Updated PPI saved to {out_path}")


def get_latest_iteration_info(base_out, ppi_base, original_ppi, cv_folds):
    """
    扫描 base_out 目录，确定当前应使用的 iteration 和 current_ppi。
    同时收集已完成迭代的 avg_score 用于恢复 best_score。
    返回: (iteration, current_ppi, best_score)
    """

    iter_dirs = glob.glob(os.path.join(base_out, 'iter_*'))
    if not iter_dirs:

        return 0, original_ppi, -1.0


    iter_indices = []
    for d in iter_dirs:
        name = os.path.basename(d)
        try:
            idx = int(name.split('_')[1])
            iter_indices.append(idx)
        except:
            continue
    if not iter_indices:
        return 0, original_ppi, -1.0

    max_i = max(iter_indices)

    next_ppi = os.path.join(ppi_base, f"{configs['ppi']}_matrix_{configs['dataset_name']}_iter{max_i+1}.csv")
    if os.path.exists(next_ppi):

        iteration = max_i + 1
        current_ppi = next_ppi
    else:

        iteration = max_i
        if max_i == 0:
            current_ppi = original_ppi
        else:
            current_ppi = os.path.join(ppi_base, f"{configs['ppi']}_matrix_{configs['dataset_name']}_iter{max_i}.csv")

            if not os.path.exists(current_ppi):
                current_ppi = original_ppi

    best_score = -1.0
    for i in range(0, iteration):
        metrics_file = os.path.join(base_out, f'iter_{i}', 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                score = data.get('avg_score', -1.0)
                if score > best_score:
                    best_score = score
    return iteration, current_ppi, best_score

def is_training_done(iter_dir, cv_folds):
    """检查训练是否已完成：所有 fold 模型文件都存在"""
    pattern = os.path.join(iter_dir, '[0-9]*.pkl')
    files = glob.glob(pattern)
    fold_numbers = set()
    for f in files:
        base = os.path.basename(f)
        parts = base.split('_')
        if parts[0].isdigit():
            fold_numbers.add(int(parts[0]))
    return len(fold_numbers) == cv_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='MCF7', choices=['MCF7', 'K562'])
    parser.add_argument('--out', type=str, default=None, help='Override output directory (default from config)')
    args = parser.parse_args()

    global configs
    configs = config_load.get()
    configs['dataset_name'] = args.dataset

    if args.out_dir is not None:
        configs['out_dir'] = args.out

    gpu = f"cuda:{args.gpu}" if args.gpu is not None else 'cpu'
    configs["device"] = gpu
    set_seed(configs['random_seed'])

    base_out = os.path.join(configs['out_dir'], args.dataset)
    os.makedirs(base_out, exist_ok=True)

    ppi_base = f"data/{configs['ppi']}"
    original_ppi = os.path.join(ppi_base, f"{configs['ppi']}_matrix.csv")


    iteration, current_ppi, best_score = get_latest_iteration_info(
        base_out, ppi_base, original_ppi, configs['cv_folds']
    )
    print(f"Resumed state: iteration={iteration}, current_ppi={current_ppi}, best_score={best_score:.4f}")

    jcma_output_dim = 20

    # ------------------- 主循环 -------------------
    while iteration < configs.get('max_iterations', 5):
        print(f"\n{'=' * 20} Iteration {iteration} {'=' * 20}")
        iter_dir = os.path.join(base_out, f'iter_{iteration}')
        os.makedirs(iter_dir, exist_ok=True)

        # --- Step 1: 加载或生成 dataset ---
        dataset_pkl = os.path.join(iter_dir, 'dataset.pkl')
        if os.path.exists(dataset_pkl):
            with open(dataset_pkl, 'rb') as f:
                dataset = pickle.load(f)
            print(f"Loaded dataset from {dataset_pkl}")
        else:
            # 仅首次（或缺失时）使用 get_data 生成
            print("Generating dataset via get_data (first time or missing)...")
            dataset = get_data(configs, ppi_path=current_ppi, load_existing=(iteration == 0))
            # 保存一份到 iter_dir，供后续直接加载
            with open(dataset_pkl, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Dataset saved to {dataset_pkl}")


        # --- Step 2: 训练（检查是否已完成） ---
        training_done = is_training_done(iter_dir, configs['cv_folds'])
        if not training_done:
            print("Training not complete, starting full training...")
            device = configs['device']
            jcma = JCMA().to(device)
            edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
            mrcgm = MRCGM(
                in_channels=jcma_output_dim,
                hidden_channels=configs['hidden_channels'],
                heads=configs['heads'],
                drop_rate=configs['drop_rate'],
                attn_drop_rate=configs['ppi_attn_drop'],
                edge_dim=edge_dim,
                residual=True,
                devices_available=device
            ).to(device)

            avg_score, _ = train_model(mrcgm, jcma, dataset, configs, iter_dir)


            metrics_path = os.path.join(iter_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump({'avg_score': avg_score}, f)
            print(f"Metrics saved to {metrics_path}")  # 新增输出，便于确认
            with open(os.path.join(iter_dir, 'log.txt'), 'a') as f:
                print(f"Iteration {iteration}: Avg Val Score (AUPRC+AUROC) = {avg_score:.4f}", file=f)


            if avg_score > best_score:
                best_score = avg_score
            else:
                if avg_score < best_score + configs.get('improvement_threshold', 0.001):
                    print(f"Performance not improved (threshold {configs.get('improvement_threshold', 0.001)}). Stopping.")
                    break
        else:
            print(f"Training already complete in {iter_dir}, skipping.")

        # --- Step 3: 加载最佳模型（用于预测和 SubgraphX） ---
        device = configs['device']
        jcma_best = JCMA().to(device)
        edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0
        mrcgm_best = MRCGM(
            in_channels=jcma_output_dim,
            hidden_channels=configs['hidden_channels'],
            heads=configs['heads'],
            drop_rate=configs['drop_rate'],
            attn_drop_rate=configs['ppi_attn_drop'],
            edge_dim=edge_dim,
            residual=True,
            devices_available=device
        ).to(device)
        mrcgm_best, jcma_best, best_ckpt = load_best_model(mrcgm_best, jcma_best, iter_dir)
        print(f"Best model loaded from {best_ckpt}")

        # --- Step 4: 全基因预测（保存 all_scores） ---
        all_scores_file = os.path.join(iter_dir, 'all_scores.pkl')
        if not os.path.exists(all_scores_file):
            print("Predicting scores for all genes...")
            all_scores = predict_all_scores_cv(mrcgm_best, jcma_best, dataset, configs, iter_dir)

            with open(all_scores_file, 'wb') as f:
                pickle.dump(all_scores, f)
            print(f"All scores saved to {all_scores_file}")
        else:
            with open(all_scores_file, 'rb') as f:
                all_scores = pickle.load(f)
            print(f"All scores loaded from {all_scores_file}")

        # --- Step 5: 生成 top-K 基因列表 ---
        top_genes_file = os.path.join(iter_dir, 'top_genes.txt')
        if not os.path.exists(top_genes_file):
            sorted_genes = sorted(all_scores, key=all_scores.get, reverse=True)
            top_genes = sorted_genes[:configs['top_k']]
            with open(top_genes_file, 'w') as f:
                for gene in top_genes:
                    f.write(gene + '\n')
            print(f"Top {len(top_genes)} genes saved to {top_genes_file}")
        else:
            with open(top_genes_file, 'r') as f:
                top_genes = [line.strip() for line in f.readlines()]
            print(f"Top genes loaded from {top_genes_file} (count: {len(top_genes)})")

        # --- Step 6: SubgraphX ---
        subgraphx_dir = os.path.join(iter_dir, 'subgraphx')
        subgraphx_pkl = os.path.join(subgraphx_dir, 'subgraphx_results.pkl')
        if not os.path.exists(subgraphx_pkl):
            print("Running SubgraphX...")

            with t.no_grad():
                fused_x = jcma_best(dataset[0].x.to(device))
            dataset[0].x = fused_x.cpu()
            sub_results = run_subgraphx_for_genes(
                gene_list=top_genes,
                model=mrcgm_best,
                dataset=dataset,
                iteration=iteration,
                top_k=configs['top_k'],
                subgraph_max_nodes=10,
                configs=configs
            )
            print(f"SubgraphX results saved to {subgraphx_dir}")
        else:
            with open(subgraphx_pkl, 'rb') as f:
                sub_results = pickle.load(f)
            print(f"SubgraphX results loaded from {subgraphx_pkl}")

        # --- Step 7: 更新 PPI 矩阵 ---
        next_ppi = os.path.join(ppi_base, f"{configs['ppi']}_matrix_{configs['dataset_name']}_iter{iteration+1}.csv")
        if not os.path.exists(next_ppi):
            print("Updating PPI matrix...")
            update_ppi_matrix(current_ppi, sub_results, next_ppi)
        else:
            print(f"Updated PPI already exists: {next_ppi}, skipping update.")
        current_ppi = next_ppi

        # --- Step 8: 用新 PPI 更新数据集并保存到下一轮 ---
        next_iter_dir = os.path.join(base_out, f'iter_{iteration+1}')
        os.makedirs(next_iter_dir, exist_ok=True)
        next_dataset_pkl = os.path.join(next_iter_dir, 'dataset.pkl')
        if not os.path.exists(next_dataset_pkl):
            print("Generating next iteration dataset by replacing PPI...")
            new_ppi_mat = pd.read_csv(next_ppi, sep='\t', index_col=0).to_numpy()
            update_dataset_ppi(dataset, new_ppi_mat, next_dataset_pkl)
        else:
            print(f"Next iteration dataset already exists: {next_dataset_pkl}, skipping.")

        # 进入下一轮
        iteration += 1

    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()