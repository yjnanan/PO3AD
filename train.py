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
from config.config_train import get_parser

# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(train_loader, model, model_fn, optimizer, epoch, max_batch_iter):
    model.train()

    # #for log the run time and remain time
    iter_time = log.AverageMeter()
    batch_time = log.AverageMeter()
    start_time = time.time()
    end_time = time.time()  # initialization
    am_dict = {}

    # #start train
    for i, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end_time)  # update time

        cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs, clip=1e-6)  # adjust lr
        loss, _, visual_dict, meter_dict = model_fn(batch, model, cfg)

        # # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #average batch loss, time for print
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = log.AverageMeter()
            am_dict[k].update(v[0], v[1])

        current_iter = (epoch-1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter
        iter_time.update(time.time() - end_time)
        end_time = time.time()
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        sys.stdout.write("epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f})  data_time: {:.2f}({:.2f}) "
                         "iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n"
                         .format(epoch, cfg.epochs, i + 1, len(train_loader), am_dict['loss'].val,
                                 am_dict['loss'].avg,
                                 batch_time.val, batch_time.avg, iter_time.val, iter_time.avg,
                                 remain_time=remain_time))
        if (i == len(train_loader) - 1): print()

    logger.info("epoch: {}/{}, train loss: {:.4f},  time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                      time.time() - start_time))
    # #write tensorboardX
    lr = optimizer.param_groups[0]['lr']
    for k in am_dict.keys():
        if k in visual_dict.keys():
            writer.add_scalar(k + '_train', am_dict[k].avg, epoch)
            writer.add_scalar('train/learning_rate', lr, epoch)

    # save pretrained model
    pretrain_file = log.checkpoint_save_newest(model, optimizer, cfg.logpath, epoch, cfg.save_freq)
    logger.info('Saving {}'.format(pretrain_file))
    pass

def SingleCard_training(cfgs):
    global cfg
    cfg = cfgs
    # logger and summary write
    global logger
    from tools.log import get_logger
    logger = get_logger(cfg)
    logger.info(cfg)  # log config
    # # summary writer
    global writer
    writer = SummaryWriter(cfg.logpath)

    logger.info('=> creating model ...')
    from network.PO3AD import PONet as net
    from network.PO3AD import model_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    model = net(cfg.in_channels, cfg.out_channels)
    model = model.cuda()

    logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    #  #optimizer
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99),
                                weight_decay=cfg.weight_decay)
    # load dataset
    if cfg.dataset == 'AnomalyShapeNet':
        from datasets.AnomalyShapeNet.dataset_preprocess import Dataset
    elif cfg.dataset == 'Real3D':
        from datasets.Real3D.dataset_preprocess import Dataset
    else:
        print('do not support this dataset at present')

    dataset = Dataset(cfg)
    dataset.trainLoader()
    logger.info('Training samples: {}'.format(len(dataset.train_file_list)))

    max_batch_iter = len(dataset.train_file_list) // cfg.batch_size

    # #train
    cfg.pretrain = ''  # Automatically identify breakpoints
    start_epoch, pretrain_file = log.checkpoint_restore(model, None, cfg.logpath, pretrain_file=cfg.pretrain)
    logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                else 'Start from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, cfg.epochs):
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch, max_batch_iter)
    pass





if __name__ == '__main__':
    cfg = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id
    # fix seed for debug
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)

    # # Determine whether it is distributed training
    SingleCard_training(cfg)