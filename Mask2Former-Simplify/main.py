#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/10/11 19:54:03
@Author  :   zzubqh 
@Version :   1.0
@Contact :   baiqh@microport.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'  # 

from fvcore.common.config import CfgNode
from configs.config import Config
import torch
torch.cuda.set_device(0)
from maskformer_train import MaskFormer
from dataset.dataset import ADE200kDataset, NuImagesDataset
from Segmentation import Segmentation
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
import glob
import tqdm
if torch.cuda.device_count() > 1:
    torch.distributed.init_process_group(backend='nccl')

def user_scattered_collate(batch):
    data = [item['images'] for item in batch]
    masks = [item['masks'] for item in batch]
    out = {'images': torch.cat(data, dim=0), 'masks': torch.cat(masks, dim=0)}
    return out

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/maskformer_ake150.yaml')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--project_name", default='NuImages_swin_base_Seg', type=str)

    args = parser.parse_args()
    cfg_ake150 = Config.fromfile(args.config)

    cfg_base = CfgNode.load_yaml_with_base(args.config, allow_unsafe=True)    
    cfg_base.update(cfg_ake150.__dict__.items())

    cfg = cfg_base
    for k, v in args.__dict__.items():
        cfg[k] = v

    cfg = Config(cfg)

    cfg.ngpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        cfg.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(cfg.local_rank)
    return cfg


def train_ade200k():
    cfg = get_args()
    dataset_train = ADE200kDataset(cfg.DATASETS.TRAIN, cfg, dynamic_batchHW=True)
    if cfg.ngpus > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, rank=cfg.local_rank)
    else:
        train_sampler = None                            
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False if train_sampler is not None else True,  
        collate_fn=dataset_train.collate_fn,
        num_workers=cfg.TRAIN.WORKERS,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)

    dataset_eval = ADE200kDataset(cfg.DATASETS.VALID, cfg)
    loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=1,
        shuffle=False,  
        collate_fn=dataset_eval.collate_fn,
        num_workers=cfg.TRAIN.WORKERS)

    seg_model = MaskFormer(cfg)
    seg_model.train(train_sampler, loader_train, loader_eval, cfg.TRAIN.EPOCH)

def train_nuimages():
    cfg = get_args()
    dataset_train = NuImagesDataset(cfg.DATASETS.ROOT_DIR, cfg, version='v1.0-train') # v1.0-mini or v1.0-train
    dataset_eval = NuImagesDataset(cfg.DATASETS.ROOT_DIR, cfg, version='v1.0-val') # v1.0-mini or v1.0-val

    if cfg.ngpus > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, rank=cfg.local_rank)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval, rank=cfg.local_rank)
    else:
        train_sampler = None     
        eval_sampler = None

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False if train_sampler is not None else True,  
        collate_fn=dataset_train.collate_fn,
        num_workers=cfg.TRAIN.WORKERS,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler)
    
    loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=1,
        shuffle=False if eval_sampler is not None else True,  
        collate_fn=dataset_eval.collate_fn,
        num_workers=cfg.TRAIN.WORKERS,
        drop_last=False,
        pin_memory=True,
        sampler=eval_sampler)

    seg_model = MaskFormer(cfg)
    seg_model.train(train_sampler, loader_train, loader_eval, cfg.TRAIN.EPOCH)

def segmentation_test():
    cfg = get_args()
    # segmentation_handler = Segmentation(cfg)
    # segmentation_handler.forward()
    # Predict and save an image
    input_dir  = "/space/hotel/bachn/implement_model/Mask2Former-Simplify/test"
    output_dir  = "/space/hotel/bachn/implement_model/Mask2Former-Simplify/output"

    segmentation = Segmentation(cfg)

    # Get list of image paths
    image_paths = glob.glob(os.path.join(input_dir, '*.[jp][pn]g'))  # Adjust the pattern as necessary

    # Process each image
    for img_path in image_paths:
        # Process image
        predicted_masks = segmentation.forward([img_path])

        # Save output mask
        img_name = os.path.basename(img_path)
        mask_output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask.png")
        # cv2.imwrite(mask_output_path, predicted_masks[0])

        # Optional: Save combined image and mask
        original_img = Image.open(img_path)
        combined_output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_combined.png")
        segmentation.render_image(original_img, predicted_masks[0], combined_output_path)
    

if __name__ == '__main__':
    # train_ade200k()
    # train_nuimages()
    segmentation_test()