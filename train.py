#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.autograd import Variable
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os
from FusionNetwork import FusionNet
from Conversion import RGB2YCrCb,YCrCb2RGB
from logger import setup_logger
from LossFunsion import fusion_loss_vif
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()
def train_fusion(logger=None):
    lr_start = 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    fusionmodel = eval('FusionNet')(output=1)
    fusionmodel.cuda()
    fusionmodel.train()
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)
    criteria_fusion = fusion_loss_vif()
    epoch = 10
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')
    for epo in range(0, epoch):
        lr_start = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it, (image_vis, image_ir, name) in enumerate(train_loader):
            fusionmodel.train()
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()
            logits = fusionmodel(image_vis_ycrcb, image_ir)
            # 融合图像的Y通道
            image_vis_y = image_vis_ycrcb[:, :1, :, :]
            optimizer.zero_grad()
            loss_total, loss_grad, loss_in, loss_ssim = criteria_fusion(
                image_vis_y, image_ir, logits)
            loss_total.backward()
            optimizer.step()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                        'lr:{lr:.4f}'
                    ]
                ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=loss_ssim,
                    time=t_intv,
                    eta=eta,
                    lr=lr_this_epo
                )
                logger.info(msg)
                st = ed
    fusion_model_file = os.path.join(modelpth, 'fusion_modelz.pth')  #
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    logger.info("Fusion Model Save to: {}".format(fusion_model_file))
    logger.info('\n')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='AcFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=2)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    train_fusion(logger)
    print("Train Fusion Model Sucessfully~!")
    print("training Done!")