import os
import copy
import glob
import pickle
import torch as t
import torch.nn
import numpy as np
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, roc_auc_score
from utils import calculate_metrics, get_gene_list
import config_load

configs = config_load.get()


def train(model, attnmodel, fea, fold, train_loader_list, valid_loader_list, optimizer, devices, scheduler=None, loss_func=None):
    attnmodel.train()
    model.train()
    tot_loss = 0
    steps = 0

    for data in zip(*train_loader_list):
        steps += 1
        optimizer.zero_grad()
        size = data[0].batch_size

        pos = data[0].pos
        if pos is not None:
            if pos.dim() == 2:
                pos = pos.view(-1)
        else:
            # 如果没有 pos，使用 n_id（备选）
            pos = data[0].n_id

        features = attnmodel(fea)
        # subgraph_features = features[data[0].pos]
        subgraph_features = features[pos]

        data[0].x = subgraph_features

        out = model(data)


        out = out[:size]
        true_lab = data[0].y[:size][:, 1].to(devices)
        out = out.view(-1)
        loss = loss_func(out, true_lab.float())
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    tot_loss = tot_loss / steps

    attnmodel.eval()
    model.eval()
    y_true = np.array([])
    y_score = np.array([])
    train_correct = 0
    num_train = 0
    for data in zip(*train_loader_list):
        size = data[0].batch_size
        with torch.no_grad():
            features = attnmodel(fea)
            subgraph_features = features[data[0].pos]
            data[0].x = subgraph_features
            out = model(data)
            out = out[:size]
        true_lab = data[0].y[:size][:, 1].to(devices)
        out = out.view(-1)
        pred_lab = t.zeros(size)
        pred_lab[out > 0.5] = 1
        pred_lab = pred_lab.to(devices)
        train_correct += t.eq(pred_lab, true_lab).sum().float()
        num_train += size
        train_mask = data[0].train_mask[:size, fold]
        if y_score.size == 0:
            y_score = out[train_mask[:size]].cpu().detach()
        else:
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
        y_true = np.append(y_true, true_lab.cpu().detach().numpy())

    train_acc = (train_correct / num_train).cpu().detach().numpy()
    train_auprc = average_precision_score(y_true=y_true, y_score=y_score)
    train_auc = roc_auc_score(y_true, y_score)
    print(f"Train AUPRC: {train_auprc:.4f}, AUROC: {train_auc:.4f}, ACC: {train_acc:.4f}")


    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])
    valid_loss = 0
    steps = 0
    for data in zip(*valid_loader_list):
        steps += 1
        size = data[0].batch_size
        with torch.no_grad():
            features = attnmodel(fea)
            subgraph_features = features[data[0].pos]
            data[0].x = subgraph_features
            out = model(data)
            out = out[:size]
        true_lab = data[0].y[:size][:, 1].to(devices)
        out = out.view(-1)
        pred_lab = t.zeros(size)
        pred_lab[out > 0.5] = 1
        pred_lab = pred_lab.to(devices)
        y_pred = np.append(y_pred, pred_lab.cpu().detach().numpy())
        if y_score.size == 0:
            y_score = out.cpu().detach()
        else:
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
        y_true = np.append(y_true, true_lab.cpu().detach().numpy())
        valid_loss += loss_func(out, true_lab.float()).item()

    valid_loss = valid_loss / steps
    acc, cf_matrix, auprc, f1, auc = calculate_metrics(y_true, y_pred, y_score)

    return tot_loss, valid_loss, acc, cf_matrix, auprc, f1, auc, train_auprc, train_auc, train_acc, y_true, y_score


import glob
import os

def train_model_original(modules, params, fold, iter_dir, head_info=False):

    device = params["device"]
    model = modules['model']
    attnmodel = modules['attnmodel']
    loader_list = modules['train_loader_list']
    valid_loader_list = modules['valid_loader_list']
    optimizer = modules['optimizer']
    scheduler = modules['scheduler']
    loss_func = modules['loss_func']
    dataset = modules['dataset']

    data = dataset[0]
    if head_info:
        with open(os.path.join(iter_dir, 'log.txt'), 'a') as f:
            print(f"Fold {fold} start: Train/Valid/Test = {data.train_mask[:, fold].sum()}, {data.valid_mask[:, fold].sum()}, {data.test_mask.sum()}", file=f)

    print('Start Training')
    vmax_auprc = 0
    trigger_times = 0
    fea = dataset[0].x
    best_model_path = None
    best_auc = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    best_tp = 0
    max_epoch = 0

    for epoch in range(params['num_epochs']):
        train_loss, valid_loss, acc, cf_matrix, auprc, f1, auc, train_auprc, train_auc, train_acc, y_true, y_score = train(
            model, attnmodel, fea, fold, loader_list, valid_loader_list,
            optimizer, device, scheduler, loss_func
        )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Acc: {acc:.4f}, "
                  f"Auprc: {auprc:.4f}, TP: {cf_matrix[1, 1]}, F1: {f1:.4f}, Auroc: {auc:.4f}")


        if epoch >= params["num_epochs"] // 10:
            if auprc < vmax_auprc:
                trigger_times += 1
                if trigger_times == params["num_epochs"] // 5:
                    print("Early Stopping")
                    break
            else:
                trigger_times = 0
                vmax_auprc = auprc
                max_epoch = epoch
                best_acc = acc
                best_tp = cf_matrix[1, 1]
                best_auc = auc
                best_f1 = f1


                for old_file in glob.glob(os.path.join(iter_dir, f"{fold}_*.pkl")):
                    os.remove(old_file)


                model_path = os.path.join(iter_dir, f"{fold}_{vmax_auprc:.4f}_{best_auc:.4f}_{best_tp}.pkl")
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'jcma_state_dict': {k: v.half() for k, v in attnmodel.state_dict().items()},
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, model_path)
                best_model_path = model_path


    dataset[0].train_mask[modules['drop_idx'], fold] = True

    return vmax_auprc, best_auc, best_acc, best_f1, best_tp, best_model_path, y_true, y_score

def load_fold_model(model, attnmodel, model_dir, fold):

    device = next(model.parameters()).device
    pattern = os.path.join(model_dir, f'{fold}_*.pkl')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint for fold {fold}")
    ckpt = sorted(files)[-1]
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    attnmodel.load_state_dict(checkpoint['jcma_state_dict'])
    return model, attnmodel


def load_best_model(model, attnmodel, model_dir):

    device = next(model.parameters()).device
    pattern = os.path.join(model_dir, '*_*.pkl')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    best_pr = -1
    best_file = None
    for f in files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 3:
            try:
                pr = float(parts[1])
                if pr > best_pr:
                    best_pr = pr
                    best_file = f
            except:
                continue
    if best_file is None:
        raise FileNotFoundError("No valid checkpoint with AUPRC found.")
    print(f"Loading best model from {best_file} with AUPRC {best_pr:.4f}")
    checkpoint = torch.load(best_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    attnmodel.load_state_dict(checkpoint['jcma_state_dict'])
    return model, attnmodel, best_file


def get_training_modules(model, attnmodel, dataset, fold, configs):

    device = configs['device']
    batch_size = configs['batch_size']
    neighbors = configs['neighbors']
    loss_func = torch.nn.BCELoss()


    def drop_samples(dataset, fold, sample_neg=0., sample_pos=1., num_samples=0, random_seed=42):
        if sample_neg == 1 and sample_pos == 1:
            return []
        import random
        drop_neg = 1 - sample_neg
        drop_pos = 1 - sample_pos
        splitted_idx = dataset.get_idx_split(fold)
        train_idx = splitted_idx['train']
        drop_neg_idx, drop_pos_idx = [], []
        for i in train_idx:
            if dataset[0].y[i][0]:
                drop_neg_idx.append(i.item())
            if dataset[0].y[i][1]:
                drop_pos_idx.append(i.item())
        num_neg_samples = len(drop_neg_idx)
        num_pos_samples = len(drop_pos_idx)
        random.seed(random_seed)
        if num_samples:
            num_neg = int(num_samples * num_neg_samples / (num_neg_samples + num_pos_samples))
            num_pos = num_samples - num_neg
            drop_neg_idx = random.sample(drop_neg_idx, num_neg_samples - num_neg)
            drop_pos_idx = random.sample(drop_pos_idx, num_pos_samples - num_pos)
        else:
            drop_neg_idx = random.sample(drop_neg_idx, int(num_neg_samples * drop_neg))
            drop_pos_idx = random.sample(drop_pos_idx, int(num_pos_samples * drop_pos))
        drop_idx = drop_neg_idx + drop_pos_idx
        print(f"Negatives: {num_neg_samples - len(drop_neg_idx)}, Positives: {num_pos_samples - len(drop_pos_idx)}")
        dataset[0].train_mask[drop_idx, fold] = False
        return drop_idx

    drop_idx = drop_samples(dataset, fold, sample_neg=configs['sample_neg'],
                            sample_pos=configs['sample_pos'], random_seed=configs['random_seed'])

    data = dataset[0]

    train_nodes = torch.nonzero(data.train_mask[:, fold]).view(-1)
    valid_nodes = torch.nonzero(data.valid_mask[:, fold]).view(-1)
    test_nodes = torch.nonzero(data.test_mask).view(-1)
    unknown_nodes = torch.nonzero(data.unlabeled_mask).view(-1)

    train_loader = NeighborLoader(data, num_neighbors=neighbors, batch_size=batch_size,
                                  directed=False, input_nodes=train_nodes, shuffle=True)
    valid_loader = NeighborLoader(data, num_neighbors=neighbors, batch_size=batch_size,
                                  directed=False, input_nodes=valid_nodes, shuffle=False)
    test_loader = NeighborLoader(data, num_neighbors=neighbors, batch_size=batch_size,
                                 directed=False, input_nodes=test_nodes, shuffle=False)
    unknown_loader = NeighborLoader(data, num_neighbors=neighbors, batch_size=batch_size,
                                    directed=False, input_nodes=unknown_nodes, shuffle=False)


    optimizer = torch.optim.AdamW([
        dict(params=model.convs.parameters(), weight_decay=configs['weight_decay'], lr=configs['lr']),
        dict(params=model.lins.parameters(), weight_decay=configs['weight_decay'], lr=configs['lr']),
        dict(params=attnmodel.parameters(), weight_decay=0.001, lr=1e-7)
    ])

    num_training_steps = data.train_mask[:, fold].sum() / batch_size * configs['num_epochs']
    warmup_steps = 0.2 * num_training_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)

    modules = {
        'dataset': dataset,
        'model': model,
        'attnmodel': attnmodel,
        'loss_func': loss_func,
        'train_loader_list': [train_loader],
        'valid_loader_list': [valid_loader],
        'test_loader_list': [test_loader],
        'unknown_loader_list': [unknown_loader],
        'optimizer': optimizer,
        'scheduler': scheduler,
        'drop_idx': drop_idx
    }
    return modules


def train_model(model, attnmodel, dataset, configs, iter_dir):
    """
    10 折交叉验证训练。
    每个 fold 都会重新初始化模型
    并保存该折的最佳模型到 iter_dir。
    """
    device = configs['device']
    num_folds = configs['cv_folds']
    all_test_y_true, all_test_y_score, all_test_y_pred, all_test_y_index = [], [], [], []
    best_scores = []

    model_class = type(model)
    attn_class = type(attnmodel)

    if hasattr(dataset[0], 'pos') and dataset[0].pos is not None:
        dataset[0].pos = dataset[0].pos.view(-1)


    jcma_output_dim = 20
    edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0

    for fold in range(num_folds):
        print(f"\n--- Fold {fold} ---")


        jcma = attn_class().to(device)
        MRCGM = model_class(
            in_channels=jcma_output_dim,
            hidden_channels=configs['hidden_channels'],
            heads=configs['heads'],
            drop_rate=configs['drop_rate'],
            attn_drop_rate=configs['ppi_attn_drop'],
            edge_dim=edge_dim,
            residual=True,
            devices_available=device
        ).to(device)


        modules = get_training_modules(MRCGM, jcma, dataset, fold, configs)


        vmax_auprc, best_auc, best_acc, best_f1, best_tp, model_path, _, _ = train_model_original(
            modules, configs, fold, iter_dir, head_info=(fold == 0)
        )
        best_scores.append(vmax_auprc + best_auc)


        model_copy = copy.deepcopy(MRCGM)
        attn_copy = copy.deepcopy(jcma)
        model_copy, attn_copy = load_fold_model(model_copy, attn_copy, iter_dir, fold)
        model_copy.eval()
        attn_copy.eval()

        test_loader = modules['test_loader_list'][0]
        full_feat = dataset[0].x.to(device)

        y_true_test, y_pred_test, y_score_test = evaluate(model_copy, attn_copy, test_loader, fold, device, full_feat)
        y_index_test = []
        for batch in test_loader:
            y_index_test.extend(batch.pos.cpu().numpy())
        y_index_test = np.array(y_index_test)

        all_test_y_true.append(y_true_test)
        all_test_y_score.append(y_score_test)
        all_test_y_pred.append(y_pred_test)
        all_test_y_index.append(y_index_test)

        test_acc, test_cf, test_auprc, test_f1, test_auroc = calculate_metrics(y_true_test, y_pred_test, y_score_test)
        test_tp = test_cf[1, 1]
        with open(os.path.join(iter_dir, 'log.txt'), 'a') as f:
            print(f"Fold {fold} best epoch: Val AUPRC {vmax_auprc:.4f}, AUROC {best_auc:.4f}, ACC {best_acc:.4f}, F1 {best_f1:.4f}, TP {best_tp}", file=f)
            print(f"  Test AUPRC {test_auprc:.4f}, AUROC {test_auroc:.4f}, ACC {test_acc:.4f}, F1 {test_f1:.4f}, TP {test_tp}", file=f)


        del model_copy, attn_copy, MRCGM, jcma
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    test_pkl_path = os.path.join(iter_dir, 'foldwise_test_scores.pkl')
    with open(test_pkl_path, 'wb') as f:
        pickle.dump({
            'y_true': all_test_y_true,
            'y_score': all_test_y_score,
            'y_pred': all_test_y_pred,
            'y_index': all_test_y_index
        }, f)
    print(f"[✓] Saved fold-wise test prediction results to {test_pkl_path}")


    avg_score = np.mean(best_scores)
    return avg_score, None


def evaluate(model, attnmodel, loader, fold, device, full_feat):
    """评估给定 loader，返回 y_true, y_pred, y_score"""
    attnmodel.eval()
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for batch in loader:
            size = batch.batch_size
            features = attnmodel(full_feat)
            subgraph_features = features[batch.pos]
            batch.x = subgraph_features
            out = model([batch])
            out = out[:size].view(-1)
            true_lab = batch.y[:size][:, 1].cpu().numpy()
            pred = (out > 0.5).int().cpu().numpy()
            score = out.cpu().numpy()
            y_true.extend(true_lab)
            y_pred.extend(pred)
            y_score.extend(score)
    return np.array(y_true), np.array(y_pred), np.array(y_score)


def predict_all_scores_cv(model, attnmodel, dataset, configs, iter_dir):
    """使用所有折的模型预测，返回每个节点的平均分数"""
    device = configs['device']
    data = dataset[0]
    gene_list = get_gene_list()['gene_name'].tolist()
    num_nodes = data.num_nodes
    num_folds = configs['cv_folds']
    all_scores = np.zeros((num_nodes, num_folds))

    for fold in range(num_folds):
        model_copy = copy.deepcopy(model)
        attn_copy = copy.deepcopy(attnmodel)
        load_fold_model(model_copy, attn_copy, iter_dir, fold)
        model_copy.eval()
        attn_copy.eval()
        with torch.no_grad():
            fused_x = attn_copy(data.x.to(device))
            temp_data = Data(x=fused_x, edge_index=data.edge_index.to(device),
                             edge_attr=data.edge_attr.to(device) if data.edge_attr is not None else None)
            scores = model_copy([temp_data])
            scores = scores.squeeze().cpu().numpy()
        all_scores[:, fold] = scores

    avg_scores = np.mean(all_scores, axis=1)
    return {gene_list[i]: avg_scores[i] for i in range(num_nodes)}
