_base_ = [
    '../mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py',
]

import os

# Compatibility shim for legacy dependencies under NumPy 2.
import numpy as np

if not hasattr(np, 'sctypes'):
    np.sctypes = dict(
        float=[np.float16, np.float32, np.float64],
        int=[np.int8, np.int16, np.int32, np.int64],
        uint=[np.uint8, np.uint16, np.uint32, np.uint64],
        complex=[np.complex64, np.complex128],
        others=[np.bool_, np.object_, np.bytes_, np.str_])
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'bool'):
    np.bool = np.bool_

crop_size = (512, 512)
num_classes = 9
dataset_type = 'SWRDDataset'
data_root = f'{os.getenv("DATASET_PATH", "/dataset")}/SWRD/steel-tube-dataset-all/yolo'

data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=num_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1182, 817),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1182, 817), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(
            img_path='images/train2021',
            seg_map_path='annotations/train2021'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(
            img_path='images/val2021',
            seg_map_path='annotations/val2021'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='mIoU'),
    visualization=dict(type='SegVisualizationHook', draw=True))
