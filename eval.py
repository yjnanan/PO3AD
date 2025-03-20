import os, sys
import time
import random
import torch
import numpy as np
import open3d as o3d
import torch.optim as optim
from math import cos, pi
from tensorboardX import SummaryWriter

import tools.log as log
from config.config_eval import get_parser
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score

def load_checkpoint(model, pretrain_file, gpu=0):
    map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
    checkpoint = torch.load(pretrain_file, map_location=map_location)
    model_dict = checkpoint['model']
    for k, v in model_dict.items():
        if 'module.' in k:
            model_dict = {k[len('module.'):]: v for k, v in model_dict.items()}
        break
    model.load_state_dict(model_dict, strict=False)

def eval(cfgs):
    global cfg
    cfg = cfgs
    from network.PO3AD import PONet as net
    from network.PO3AD import eval_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    model = net(cfg.in_channels, cfg.out_channels)
    model = model.cuda()
    load_checkpoint(model, cfg.logpath + cfg.checkpoint_name)

    if cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import Dataset
        gt_mask_path = f'datasets/AnomalyShapeNet/dataset/pcd/{cfg.category}/GT/'
        tag = 'positive'
    elif cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import Dataset
        gt_mask_path = f'datasets/Real3D/Real3D-AD-PCD/{cfg.category}/gt/'
        tag = 'good'
    else:
        print('do not support this dataset at present')

    dataset = Dataset(cfg)
    dataset.testLoader()
    print(f'Test samples: {len(dataset.test_file_list)}')

    model.eval()
    label_score = []
    gt_masks = []
    pred_masks = []
    for i, batch in enumerate(dataset.test_data_loader):
        torch.cuda.empty_cache()
        sample_name = batch['fn'][0].split('/')[-1].split('.')[0]

        if tag in sample_name:
            gt_masks.append(np.zeros(batch['xyz_original'].shape[0]))
        else:
            if cfg.dataset == 'AnomalyShapeNet':
                gt_mask = np.loadtxt(gt_mask_path + sample_name + '.txt', delimiter=',')[:, 3:].squeeze(1)
            elif cfg.dataset == 'Real3D':
                gt_mask = np.loadtxt(gt_mask_path + sample_name + '.txt')[:, 3:].squeeze(1)
            gt_masks.append(gt_mask)
        score, pred_mask = eval_fn(batch, model)
        pred_mask = pred_mask.detach().cpu().abs().sum(dim=-1).numpy()
        pred_masks.append(pred_mask)
        label_score += list(zip(batch['labels'].numpy().tolist(), [score.item()]))
    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    auc_roc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)
    point_pred = np.concatenate(pred_masks, axis=0)
    point_pred = (point_pred - np.min(point_pred)) / (np.max(point_pred) - np.min(point_pred))
    point_auc_roc = roc_auc_score(np.concatenate(gt_masks, axis=0), point_pred)
    point_auc_pr = average_precision_score(np.concatenate(gt_masks, axis=0), point_pred)
    print(f'object AUC-ROC: {auc_roc}, point AUC-ROC: {point_auc_roc}, object AUCP-PR: {auc_pr}, point AUCP-PR: {point_auc_pr}')



if __name__ == '__main__':
    cfg = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    eval(cfg)