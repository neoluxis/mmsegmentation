_base_ = [
    '../segformer/segformer_mit-b4_8xb2-160k_ade20k-512x512.py',
    '../_base_/datasets/swrd.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://mmassets.onedl.ai/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=9),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
