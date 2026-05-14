_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/swrd.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=9),
    auxiliary_head=dict(num_classes=9),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
