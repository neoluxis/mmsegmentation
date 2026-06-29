# SegFormer 缺陷分割模型评估指标分析

## 任务要求指标
根据 `tasks.pdf`：
- **交并比 IoU > 0.7**（个别缺陷类别）
- **像素准确率 AP > 0.75**（总体准确率）
- **平均交并比 mIoU > 0.8**（所有类别平均）

## 当前配置状态

### 1. 已有指标（在scalars.json中）
从 `/workspace/2026/grad/code/work_dirs.old/segformer_mit-b0_4xb2-40k_swrd-512x512_gaussian/20260513_003114/vis_data/scalars.json` 可以看到：
- ✅ `aAcc`: 已记录（总体准确率）- 示例：97.5, 98.36
- ✅ `mIoU`: 已记录（平均IoU）- 示例：22.5, 35.65
- ✅ `mAcc`: 已记录（平均类别准确率）- 示例：27.0, 50.15
- ✅ 训练loss和分割准确率

### 2. 缺失指标
❌ **个别类别的 IoU 值**（无法单独评估每种缺陷）
❌ **类别级别的 Dice/F-score**（可选但有帮助）
❌ **详细的每类准确率对比**

## IoUMetric 支持的指标类型

```python
# 当前配置
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

# 支持的 iou_metrics 参数：
# - 'mIoU': 计算 aAcc, IoU, Acc
# - 'mDice': 计算 Dice, Acc  
# - 'mFscore': 计算 Fscore, Precision, Recall
```

## 推荐改进方案

### 方案1：扩展评估指标（推荐）
修改配置文件中的 `val_evaluator` 和 `test_evaluator`：

```python
val_evaluator = dict(
    type='IoUMetric', 
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    nan_to_num=0)
test_evaluator = val_evaluator
```

**优势**：
- 获得完整的Dice和F-score指标
- 能够评估模型的precision和recall
- 提供多角度的性能评估

### 方案2：添加自定义Logger记录每类IoU
在训练脚本中使用回调函数记录每类的IoU值到scalar中。

## 指标含义

| 指标 | 含义 | 当前值 | 目标值 |
|------|------|--------|--------|
| **aAcc** | 总体像素准确率 | 98.36 | > 0.75 ✓ |
| **mIoU** | 平均交并比 | 35.65% | > 0.80 ✗ |
| **mAcc** | 平均类别准确率 | 50.15% | - |
| **IoU (per class)** | 每类交并比 | - | > 0.70 |
| **Dice** | Dice系数 | - | 可选 |

## 数据集信息
- **缺陷类别数**: 9 类
- **数据集**: SWRD（螺旋焊接管道X射线图像）
- **缺陷类型**: 气孔、夹渣、裂纹、咬边、未熔合、未焊透等

## 后续步骤
1. ✅ 修改 `/mmsegmentation/configs/_base_/datasets/swrd.py`
2. ✅ 更新 val_evaluator 和 test_evaluator 配置
3. ✅ 重新训练模型或运行验证
4. ✅ 检查新的scalar输出中是否包含详细指标
