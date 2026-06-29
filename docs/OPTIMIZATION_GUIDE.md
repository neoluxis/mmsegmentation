# SegFormer 性能优化完整指南

## 📊 当前状态分析

从验证指标图表看：
- **aAcc**: 98.36% ✅ 满足 >0.75
- **mIoU**: ~70% ❌ 需达到 >0.8（还差 10%）
- **mAcc**: ~83% ✅ 接近目标

**瓶颈**: mIoU 还差 10 个百分点才能满足任务要求

---

## 🎯 优化策略分层

### 第一层：参数量升级（最有效）⭐⭐⭐

#### 为什么升级参数量有效？
- **mit-b0**: 3.7M 参数 → 基础性能
- **mit-b2**: 24.5M 参数 → +3-5% mIoU（预期）
- **mit-b3**: 47.2M 参数 → +4-6% mIoU（需要 GPU 内存充足）

| 升级方案 | 预期 mIoU | 训练时间 | GPU内存 | 推荐度 |
|---------|---------|---------|--------|--------|
| mit-b0 (当前) | 70% | 4h | 6GB | 基准 |
| **mit-b2** | **73-75%** | **8h** | **10GB** | ⭐ 推荐 |
| mit-b3 | 74-76% | 12h | 16GB | 可选 |

---

## 🚀 快速开始：升级到 mit-b2

### 方案 A：仅升级参数量（简单快速）
```bash
cd /home/neolux/workspace/2026/grad/code

python mmsegmentation/tools/train.py \
    mmsegmentation/configs/swrd/segformer_mit-b2_4xb2-80k_swrd-512x512.py \
    --work-dir work_dirs/segformer_mit-b2 \
    --gpu-id 0 1
```

**预期结果**:
- 训练时间: ~8 小时
- 目标 mIoU: **73-75%**（还需其他优化达 80%）

---

### 方案 B：参数量 + 更多优化（更好效果）

如果方案 A 的 mIoU 仍未达 80%，继续执行：

#### 步骤 1：添加数据增强

编辑: `/mmsegmentation/configs/_base_/datasets/swrd.py`

找到 `train_pipeline` 的 `dict(type='Pad'...` 之前，添加：

```python
dict(type='RandomCrop', crop_size=(512, 512)),  # 随机裁剪
dict(type='RandomFlip', prob=0.5),  # 水平翻转
dict(type='PhotoMetricDistortion'),  # 光度畸变（亮度、对比度）
```

#### 步骤 2：组合损失函数

编辑: `/mmsegmentation/configs/swrd/segformer_mit-b2.py`

修改 `decode_head` 的 `loss_decode`：

```python
decode_head=dict(
    ...
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8),
        dict(type='DiceLoss', loss_weight=0.2),  # ← 新增
    ],
    ...
)
```

#### 步骤 3：提升学习率进一步

编辑: `/mmsegmentation/configs/swrd/segformer_mit-b2_4xb2-80k_swrd-512x512.py`

```python
optimizer=dict(
    type='AdamW', 
    lr=0.00012,  # ← 改为 0.00012
    ...
)
```

---

## 📋 完整优化优先级

| 优化 | 预期提升 | 难度 | 建议顺序 |
|------|---------|------|---------|
| 1. 参数量升级（mit-b2） | +2-4% | ⭐ | **第一步** |
| 2. 训练轮数 80k | +1% | ⭐ | 已包含 |
| 3. 学习率调整 | +0.5% | ⭐ | **第二步** |
| 4. 数据增强 | +1-2% | ⭐⭐ | **第三步** |
| 5. 组合损失函数 | +1-3% | ⭐⭐ | **第四步** |
| 6. 参数量再升级 (mit-b3) | +1-2% | ⭐⭐ | 如需最高精度 |

**累计预期**: 70% → 73-75% (mit-b2) → 75-77% (数据增强) → 77-80% (组合损失) → **80%+** ✓

---

## 🔧 具体优化步骤详解

### 优化 1：参数量升级 ✅ 已为您创建配置

已创建两个新配置文件：
- `segformer_mit-b2.py` - mit-b2 基础模型定义
- `segformer_mit-b2_4xb2-80k_swrd-512x512.py` - 训练配置

**改动内容**:
```python
# backbone 改动
embed_dims=64              # 从 32 → 64
num_layers=[3, 4, 6, 3]   # 从 [2,2,2,2] → [3,4,6,3]

# decode_head 改动
in_channels=[64, 128, 320, 512]  # 对应新的通道数

# 训练参数改动
lr=0.0001                   # 从 0.00006 → 0.0001
max_iters=80000            # 从 40000 → 80000
```

---

### 优化 2：数据增强配置示例

编辑 `/mmsegmentation/configs/_base_/datasets/swrd.py`

查找原始的 train_pipeline：
```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Pad', size=crop_size, pad_val=dict(img=(123.675, 116.28, 103.53), seg=255)),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='Normalize', **img_norm_cfg)
]
```

改为：
```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),  # 先缩放
    dict(type='RandomCrop', crop_size=(512, 512)),  # 随机裁剪到 512x512
    dict(type='RandomFlip', prob=0.5),  # 50% 概率水平翻转
    dict(type='PhotoMetricDistortion'),  # 光度畸变
    dict(type='Pad', size=(512, 512), pad_val=dict(img=(123.675, 116.28, 103.53), seg=255)),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='Normalize', **img_norm_cfg)
]
```

---

### 优化 3：多损失函数组合

首先安装 mmsegmentation 中的 DiceLoss（通常已内置）：

编辑 `/mmsegmentation/configs/swrd/segformer_mit-b2.py`

```python
decode_head=dict(
    type='SegformerHead',
    ...
    loss_decode=[
        dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.8,
        ),
        dict(
            type='DiceLoss',  # Dice 系数损失
            loss_weight=0.2,
        )
    ],
    ...
)
```

**工作原理**: 
- CrossEntropyLoss：关注像素级分类准确度
- DiceLoss：关注类别间平衡，特别针对小目标缺陷

---

## ⚠️ 注意事项

### GPU 内存
- **mit-b0**: ~6GB 内存
- **mit-b2**: ~10GB 内存
- **mit-b3**: ~16GB 内存

如果显存不足，可以降低 batch_size：
```python
train_dataloader = dict(batch_size=1, num_workers=4)  # 从 2 → 1
```

### 预训练权重
配置中的预训练权重 URL 可能不可用。如果下载失败：
```python
# 改为随机初始化
checkpoint = None  # 移除这一行或设为 None
```

### 训练时间
- mit-b2 + 80k 迭代 ≈ 8 小时
- 建议使用 `screen` 或 `tmux` 后台运行：
```bash
screen -S segformer_train
# 或
tmux new-session -d -s segformer_train
tmux send-keys -t segformer_train "cd /home/neolux/workspace/2026/grad/code && python ..." Enter
```

---

## 📊 监测训练进度

实时查看 mIoU：
```bash
# 查看最新的验证指标
tail -f work_dirs/segformer_mit-b2/20*.log
# 或
cat work_dirs/segformer_mit-b2/vis_data/scalars.json | grep -o '"mIoU/mIoU":[^,}]*'
```

---

## 🎯 目标达成检查清单

优化完成后，验证：

- [ ] aAcc > 0.75 (现在: 98% ✅)
- [ ] mIoU > 0.80 (现在: 70% ❌)
- [ ] 每个缺陷类的 IoU > 0.7（需查看详细指标）

如果仍未达目标，请检查：
1. 各缺陷类是否存在严重不平衡（可用加权损失）
2. 数据质量（可能需要数据清洗）
3. 考虑集成学习（多个模型投票）

---

## 🚀 推荐执行顺序

```
第一步：mit-b2 基础训练（8小时）
  ↓
检查 mIoU（期望 73-75%）
  ↓
第二步：添加数据增强（2小时）
  ↓
检查 mIoU（期望 75-77%）
  ↓
第三步：组合损失函数（1.5小时）
  ↓
最终检查 mIoU（期望 77-80%+）
  ↓
如需 80.5%+：考虑 mit-b3（需更多 GPU 内存）
```

---

## 📞 快速参考命令

```bash
# 1. 启动 mit-b2 训练
python mmsegmentation/tools/train.py \
    mmsegmentation/configs/swrd/segformer_mit-b2_4xb2-80k_swrd-512x512.py \
    --work-dir work_dirs/segformer_mit-b2 \
    --gpu-id 0 1

# 2. 验证模型
python mmsegmentation/tools/test.py \
    mmsegmentation/configs/swrd/segformer_mit-b2_4xb2-80k_swrd-512x512.py \
    work_dirs/segformer_mit-b2/latest.pth

# 3. 查看详细指标
python -c "import json; d=json.load(open('work_dirs/segformer_mit-b2/vis_data/scalars.json')); \
import pprint; pprint.pprint([v for k,v in d.items() if 'mIoU' in k][-1])"
```

---

## 📈 预期效果曲线

```
mIoU
  |
80%+ ├─────────────────────────── 目标达成 ✓
  |  │
78% ├─ mit-b3 + 数据增强 + 组合损失
  |  │
76% ├─ mit-b2 + 数据增强 + 组合损失
  |  │
74% ├─ mit-b2 + 数据增强
  |  │
73% ├─ mit-b2 基础
  |  │
70% ├─ mit-b0 当前 ◄─ 起点
  |  │
  +──┴────────────────────────────
     40k   80k   120k  160k
```

