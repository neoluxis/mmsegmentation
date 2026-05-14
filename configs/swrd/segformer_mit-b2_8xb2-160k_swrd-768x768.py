_base_ = ['./segformer_mit-b2_8xb2-160k_swrd-512x512.py']

crop_size = (768, 768)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512)))

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
    dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1182, 817), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
