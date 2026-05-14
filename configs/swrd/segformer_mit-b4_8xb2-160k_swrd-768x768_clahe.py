_base_ = ['./segformer_mit-b4_8xb2-160k_swrd-768x768.py']

crop_size = (768, 768)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CLAHE', clip_limit=2.0, tile_grid_size=(8, 8)),
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
    dict(type='CLAHE', clip_limit=2.0, tile_grid_size=(8, 8)),
    dict(type='Resize', scale=(1182, 817), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
