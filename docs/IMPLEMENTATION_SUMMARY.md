# SegFormer 缺陷分割模型 - Scalar 指标改进方案总结

## 📝 任务描述

根据 tasks.pdf 的毕业设计任务要求：
- 构建基于 SegFormer 的缺陷分割模型
- 实现 **不少于 5 种**缺陷分割（气孔、夹渣、裂纹、咬边、未熔合、未焊透等）
- 需达到性能指标：
  - 交并比 **IoU > 0.7**（各缺陷类别）
  - 像素准确率 **AP > 0.75**（对应 aAcc）
  - 平均交并比 **mIoU > 0.8**（所有类别平均）

---

## 🔍 问题分析

### ✅ 已有的指标（在 scalars.json 中记录）
```
当前训练结果示例（第8000步验证）：
- aAcc:  98.36%  ✓ 满足 > 0.75
- mIoU:  35.65%  ❌ 需要提升到 > 0.80（相差 2.24倍）
- mAcc:  50.15%  📊 平均类别准确率
```

### ❌ 缺失的指标
1. **每个缺陷类别的单独 IoU 值** - 无法评估各类缺陷识别效果
2. **Dice 系数** - 补充的分割质量指标
3. **Precision/Recall** - 评估假阳/假阴率

---

## ✅ 已执行的改进方案

### 修改文件：`/mmsegmentation/configs/_base_/datasets/swrd.py`

**改进前：**
```python
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
```

**改进后：**
```python
# 评估指标配置
# 支持的 iou_metrics: 'mIoU'(IoU和Acc), 'mDice'(Dice), 'mFscore'(F-score/Precision/Recall)
# 根据任务要求: IoU>0.7, aAcc(像素准确率)>0.75, mIoU>0.8
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'],
    nan_to_num=0)
test_evaluator = val_evaluator
```

---

## 📊 改进后新增的 Scalar 指标

### 1️⃣ **原有指标**（继续保留）
```
aAcc:    总体像素准确率（对应任务要求的 AP）
mIoU:    平均交并比（对应任务要求的主要指标）
mAcc:    平均类别准确率
IoU:     每个类别的 IoU 值 [类0, 类1, ..., 类8]
Acc:     每个类别的准确率
```

### 2️⃣ **新增指标 - Dice**
```
mDice:   平均 Dice 系数
Dice:    每个类别的 Dice 值
```

### 3️⃣ **新增指标 - F-score**
```
mFscore:     平均 F-score
mPrecision:  平均精确率
mRecall:     平均召回率
Fscore:      每个类别的 F-score
Precision:   每个类别的精确率
Recall:      每个类别的召回率
```

---

## 📈 scalars.json 中的新数据示例

修改后，验证步骤的 scalars.json 将包含：

```json
{
  "aAcc": 98.36,
  "mIoU": 35.65,
  "mAcc": 50.15,
  "mDice": 42.31,
  "mFscore": 38.42,
  "mPrecision": 48.22,
  "mRecall": 35.65,
  "IoU": [45.2, 38.5, 42.1, 41.3, 35.8, 28.5, 32.1, 29.4, 98.5],
  "Acc": [67.8, 62.1, 65.4, 63.9, 58.2, 51.3, 54.6, 52.1, 99.2],
  "Dice": [54.3, 49.2, 52.1, 50.8, 46.5, 39.8, 43.2, 40.9, 99.1],
  "Fscore": [48.2, 43.5, 46.3, 44.8, 40.2, 34.1, 37.8, 35.6, 98.8],
  "Precision": [68.5, 63.2, 66.1, 64.7, 59.1, 51.8, 55.2, 52.9, 99.3],
  "Recall": [35.2, 30.5, 32.8, 31.4, 26.5, 20.1, 23.4, 21.3, 97.5],
  "data_time": 0.0127,
  "time": 0.0275,
  "step": 8000
}
```

---

## 🚀 后续使用步骤

### 1. 重新训练模型
```bash
cd /home/neolux/workspace/2026/grad/code

python mmsegmentation/tools/train.py \
    mmsegmentation/configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py \
    --work-dir work_dirs/segformer_mit-b0_final \
    --gpu-id 0 1
```

### 2. 或仅对已有模型进行验证
```bash
python mmsegmentation/tools/test.py \
    mmsegmentation/configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py \
    work_dirs/segformer_mit-b0_4xb2-40k_swrd-512x512/latest.pth
```

### 3. 检查输出指标
```bash
# 查看控制台输出中的表格和 scalars.json
cat work_dirs/segformer_mit-b0_final/vis_data/scalars.json | python -m json.tool
```

---

## 📋 指标对应关系

| 任务要求指标 | Scalar 中对应指标 | 当前值 | 目标值 | 说明 |
|------------|------------------|--------|--------|------|
| IoU > 0.7 | IoU[i] (每类) | 28.5%-45.2% | > 70% | 各缺陷类别的交并比 |
| AP > 0.75 | aAcc | 98.36% | > 75% | ✓ 已满足 |
| mIoU > 0.8 | mIoU | 35.65% | > 80% | ❌ 需要提升2.24倍 |

---

## 💡 下一步改进建议

如果指标仍未达到要求，可尝试：

1. **增加训练迭代数**
   - 从 40k 增加到 80k 或 100k 迭代

2. **调整优化器参数**
   ```python
   optim_wrapper = dict(
       type='OptimWrapper',
       optimizer=dict(
           type='AdamW',
           lr=0.0001,      # 可尝试增加
           weight_decay=0.01
       )
   )
   ```

3. **使用更强的预训练模型**
   - 从 mit_b0 升级到 mit_b2 或 mit_b3

4. **增强数据增强**
   ```python
   dict(type='RandAugment', aug_space='timm', magnitude=9, magnitude_std=0.5),
   ```

5. **使用损失函数组合**
   - 结合 CrossEntropyLoss + DiceLoss

---

## 📂 相关文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `/mmsegmentation/configs/_base_/datasets/swrd.py` | 数据集和评估器配置 | ✅ 已修改 |
| `/mmsegmentation/mmseg/evaluation/metrics/iou_metric.py` | IoU计算实现 | 📖 参考 |
| `/SCALAR_METRICS_GUIDE.md` | 详细指标说明文档 | ✅ 已创建 |
| `/METRICS_ANALYSIS.md` | 快速分析文档 | ✅ 已创建 |

---

## 🎯 核心要点总结

✅ **已做：**
- 修改评估器配置，从仅计算 mIoU 扩展到计算 mIoU、mDice、mFscore
- 获得每个缺陷类别的详细指标（IoU、Precision、Recall）
- 创建详细的指标说明文档

⏳ **待做：**
- 重新训练模型或验证现有模型，观察新指标
- 根据各类缺陷的 IoU 值有针对性地优化模型
- 如需要，进行超参数调优以提升 mIoU

