#!/usr/bin/python3
#coding=utf-8

from functools import reduce
import os
import sys

#sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from skimage import img_as_ubyte, img_as_float
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from data import dataset
import time
import logging as logger
import json
import subprocess
GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
#os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

TAG = "test"
SAVE_PATH = TAG
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="test_%s.log"%('tmp'), filemode="w")

root = './CodDataset'
DATASETS = [f'{root}/test/CAMO',f'{root}/test/CHAMELEON',f'{root}/test/COD10K',f'{root}/test/HCK4',]
device = torch.device("cuda:2")



def tensor_to_bgr(img_tensor, use_imagenet_norm=True):
    """
    img_tensor: [1, 3, H, W] or [3, H, W]
    return: BGR uint8 image for cv2
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]

    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()  # HWC, RGB

    # 如果你的 dataset 里用了 ImageNet 归一化，就保留这段
    if use_imagenet_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img * std + mean

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def make_highlight_image(img_bgr, pred_mask, bg_alpha=0.25, thresh=0.5, draw_contour=True):
    """
    img_bgr: 原图(BGR, uint8)
    pred_mask: 预测前景图 [H, W], 值域[0,1]
    bg_alpha: 背景保留亮度，越小背景越暗
    """
    pred_mask = np.clip(pred_mask, 0, 1).astype(np.float32)

    # 前景保持亮，背景压暗
    mask3 = pred_mask[..., None]  # [H, W, 1]
    highlight = img_bgr.astype(np.float32) * (bg_alpha + (1.0 - bg_alpha) * mask3)
    highlight = np.clip(highlight, 0, 255).astype(np.uint8)

    # 可选：画前景轮廓
    if draw_contour:
        bin_mask = (pred_mask >= thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlight, contours, -1, (0, 255, 0), 2)

    return highlight










class DC(nn.Module):
    def __init__(self, h):
        super().__init__()
        w = h.weight
        b = h.bias
        self.w = w
        self.c = w.shape[1]
        self.b = b
    def forward(self, x):
        c1 = F.conv2d(x, self.w.transpose(0,1), padding=(1,1), groups=self.c)
        return c1.sum(1, keepdims=True) + self.b

class Test(object):
    def __init__(self, Dataset, datapath, Network):
        ## dataset
        self.datapath = datapath.split("/")[-1]
        print("Testing on %s"%self.datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network
        self.net.train(False)
        self.net.to(device)
        self.net.eval()

    def accuracy(self, map_path=None):
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0

            for image, mask, shape, maskpath in self.loader:
                start_time = time.time()
                if map_path is None:
                    image, mask            = image.to(device).float(), mask.to(device).float()
                    out2, attn_map = self.net(image, shape)
                    """by_chf:"""
                    #out2 = out2 #+ attn_map

                    """*****"""
                    pred                   = torch.sigmoid(out2)
                    torch.cuda.synchronize()
                else:
                    file_name = maskpath[0].split('/')[-1].split('.')[0]
                    dataset_name = self.datapath
                    pred_name = os.path.join(map_path, dataset_name, file_name+'.png')
                    # read img as float
                    pred = cv2.imread(pred_name, cv2.IMREAD_GRAYSCALE)
                    pred = img_as_float(pred)
                    pred = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)

                end_time = time.time()
                cost_time += end_time - start_time

                ## MAE
                cnt += 1
                mae += (pred-mask).abs().mean()
                ## F-Score
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred >= threshod[i]).float()
                    precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)
                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
                if cnt % 20 == 0:
                    fps = image.shape[0] / (end_time - start_time)
                    # print('MAE=%.6f, F-score=%.6f, fps=%.4f'%(mae/cnt, fscore.max()/cnt, fps))
            fps = len(self.loader.dataset) / cost_time
            msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f'%(self.datapath, mae/cnt, fscore.max()/cnt, len(self.loader.dataset), fps)
            print(msg)
            logger.info(msg)

    def save(self):
        with torch.no_grad():
            cost_time = 0
            cnt = 0
            mae = 0
            print('will save to ./map/{}'.format(EXP_NAME))

            dataset_name = self.cfg.datapath.split('/')[-1]

            # 1. 原来的灰度预测图
            head_mask = './map/{}/{}'.format(EXP_NAME, dataset_name)
            # 2. 新增：前景高亮图
            head_highlight = './map/{}_highlight/{}'.format(EXP_NAME, dataset_name)
            # 3. 可选：透明前景图
            head_rgba = './map/{}_rgba/{}'.format(EXP_NAME, dataset_name)

            os.makedirs(head_mask, exist_ok=True)
            os.makedirs(head_highlight, exist_ok=True)
            os.makedirs(head_rgba, exist_ok=True)

            for image, mask, (H, W), name in self.loader:
                start_time = time.perf_counter()

                image_cuda = image.to(device).float()
                out2, out_dst = self.net(image_cuda, (H, W))

                torch.cuda.synchronize()
                cost_time += time.perf_counter() - start_time

                # ---------- 预测图 ----------
                pred = torch.sigmoid(out2[0, 0]).detach().cpu()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

                # 若网络输出尺寸和原图不同，插值到原图尺寸
                h = int(H) if not torch.is_tensor(H) else int(H.item())
                w = int(W) if not torch.is_tensor(W) else int(W.item())

                if pred.shape[0] != h or pred.shape[1] != w:
                    pred = F.interpolate(
                        pred.unsqueeze(0).unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    )[0, 0]

                mae += (pred - mask).abs().mean()
                cnt += len(image)

                pred_np = pred.numpy()

                # ---------- 保存原灰度图 ----------
                cv2.imwrite(
                    os.path.join(head_mask, name[0]),
                    img_as_ubyte(pred_np)
                )

                # ---------- 还原原图 ----------
                # 如果你的 DataLoader 里没有做 ImageNet norm，把 use_imagenet_norm=False
                img_bgr = tensor_to_bgr(image, use_imagenet_norm=True)

                if img_bgr.shape[:2] != pred_np.shape:
                    pred_np = cv2.resize(pred_np, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

                # ---------- 生成前景高亮图 ----------
                highlight = make_highlight_image(
                    img_bgr=img_bgr,
                    pred_mask=pred_np,
                    bg_alpha=0.20,      # 背景更暗一点
                    thresh=0.5,
                    draw_contour=True
                )

                cv2.imwrite(
                    os.path.join(head_highlight, name[0]),
                    highlight
                )

                # ---------- 可选：保存透明前景PNG ----------
                rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = (pred_np * 255).astype(np.uint8)  # alpha通道
                png_name = os.path.splitext(name[0])[0] + '.png'
                cv2.imwrite(
                    os.path.join(head_rgba, png_name),
                    rgba
                )

            fps = len(self.loader.dataset) / cost_time
            msg = '%s len(imgs)=%s, fps=%.4f' % (self.datapath, len(self.loader.dataset), fps)
            print('mae {}'.format(mae / cnt))
            print(msg)
            logger.info(msg)
        
    def change_json(self, json_path=None):
        js = json.load(open(json_path))
        k1 = next(iter(js.values()))
        # print(self.datapath)
        # print('*****')
        #k1[self.datapath]['path'] = os.path.abspath('./map/{}/'.format(EXP_NAME) + self.datapath)
        json.dump(js, open(json_path, 'w'), indent=4)

def cal_cod_metrics(js_m, js_d):
    js_m = os.path.abspath(js_m)
    js_d = os.path.abspath(js_d)
    os.chdir('./PySODEvalToolkit')
    os.system('python ./eval.py --method {} --dataset {} --record-txt ./results.txt'.format(js_m, js_d))
    os.chdir('../')



import os
from net import Net
from pathlib import Path as pa
EXP_NAME='trained'
JSON_METHOD = './PySODEvalToolkit/cod_method.json'
JSON_DATA = './PySODEvalToolkit/cod_dataset.json'
from ptflops import get_model_complexity_info
if __name__=='__main__':
    # set torch cuda device
    cfg = dataset.Config(datapath='000', mode='test')
    net = Net(cfg)
    """by-chf"""
    device = torch.device("cuda:0")
    # net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count()))).to(device)
    # device = torch.device("cuda:1")
    net = torch.nn.DataParallel(net, device_ids=[0]).to(device)
    """****"""
    path = '/home/dell/CV408/hb/SCOD-S/scod-s/out_SSS/trained/model-best.pth'
    state_dict = torch.load(path)

    # print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))
    net.load_state_dict(state_dict, strict=True)
    # """by_chf"""
    # pretrained_dict = torch.load(path)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net.state_dict()}
    # net.load_state_dict(pretrained_dict)
    # """*****"""
    print('complete loading: {}'.format(path))
    print('-----------------')
    print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))
    macs, params = get_model_complexity_info(
        net,
        (3, 512, 512),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("*Done...")
    for e in DATASETS[:]:
        t =Test(dataset, e, net)
        # t.accuracy()
        t.save()
        t.change_json(JSON_METHOD)
    
    cal_cod_metrics(JSON_METHOD, JSON_DATA)
    print(EXP_NAME)
    #N*C x C*N