# 关于 Scalar 指标的完整指南

## 📋 任务要求指标

根据毕业设计任务书 (tasks.pdf)，需要达到以下性能指标：
- **交并比 IoU > 0.7** （各缺陷类别）
- **像素准确率 AP > 0.75** （总体准确率，对应 aAcc）
- **平均交并比 mIoU > 0.8** （所有类别平均IoU）
- **缺陷类别数 ≥ 5** （气孔、夹渣、裂纹、咬边、未熔合、未焊透等）

---

## ✅ 当前状态分析

### 已有的 Scalar 指标（从训练日志中）

文件：`/workspace/2026/grad/code/work_dirs.old/segformer_mit-b0_4xb2-40k_swrd-512x512_gaussian/20260513_003114/vis_data/scalars.json`

**示例数据（第8000步验证）：**
```json
{
  "aAcc": 98.36,        ✅ 总体像素准确率（AP）- 满足 >0.75
  "mIoU": 35.65,        ❌ 平均IoU - 需要提升到 >0.80
  "mAcc": 50.15,        📊 平均类别准确率
  "data_time": 0.0127,  ⏱️ 数据加载时间
  "time": 0.0275,       ⏱️ 处理时间
  "step": 8000          📍 当前步数
}
```

### 缺失的指标

❌ **每个类别的单独 IoU 值** - 无法评估各缺陷识别效果
❌ **Dice 系数** - 用于评估分割质量的补充指标
❌ **Precision/Recall** - 用于评估假阳/假阴率

---

## 🔧 解决方案：扩展 Scalar 指标

### 修改位置
文件：`/mmsegmentation/configs/_base_/datasets/swrd.py`

### 修改前（原始配置）
```python
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
```

### 修改后（扩展配置）✅
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

## 📊 修改后会获得的新指标

### 1. **mIoU 类相关指标（已有）**
```
aAcc:  总体像素准确率
IoU:   [类别0的IoU, 类别1的IoU, ..., 类别8的IoU]
Acc:   [类别0的准确率, 类别1的准确率, ..., 类别8的准确率]
mIoU:  所有类别IoU的平均值（这是主要指标）
mAcc:  所有类别准确率的平均值
```

### 2. **mDice 类相关指标（新增）**
```
Dice:  [类别0的Dice, 类别1的Dice, ..., 类别8的Dice]
mDice: 所有类别Dice的平均值
```

### 3. **mFscore 类相关指标（新增）**
```
Fscore:    [类别0的F-score, 类别1的F-score, ..., 类别8的F-score]
Precision: [类别0的精确率, 类别1的精确率, ..., 类别8的精确率]
Recall:    [类别0的召回率, 类别1的召回率, ..., 类别8的召回率]
mFscore:   所有类别F-score的平均值
mPrecision: 平均精确率
mRecall:   平均召回率
```

---

## 📈 指标解释表

| 指标 | 公式 | 含义 | 任务要求 | 当前值 |
|------|------|------|--------|--------|
| **IoU** (交并比) | $\frac{TP}{TP+FP+FN}$ | 预测正确的缺陷像素占总缺陷像素的比例 | > 0.7 | 35.65%→需提升 |
| **aAcc** (像素准确率) | $\frac{TP+TN}{全部像素}$ | 所有正确分类的像素比例 | > 0.75 | 98.36% ✓ |
| **mIoU** (平均IoU) | $\frac{\sum IoU_i}{类别数}$ | 所有缺陷类别IoU的平均 | > 0.80 | 35.65% ❌ |
| **Acc** (类准确率) | $\frac{TP}{TP+FN}$ | 该类别被正确识别的比例 | - | 50.15% |
| **Dice** 系数 | $\frac{2TP}{2TP+FP+FN}$ | 分割重叠程度（对小物体敏感） | - | 新增 |
| **Precision** | $\frac{TP}{TP+FP}$ | 预测为缺陷的正确率 | - | 新增 |
| **Recall** | $\frac{TP}{TP+FN}$ | 缺陷被检出的比例 | - | 新增 |

### 符号说明
- **TP** (True Positive): 正确识别的缺陷像素
- **FP** (False Positive): 错误识别的背景像素
- **FN** (False Negative): 漏识别的缺陷像素
- **TN** (True Negative): 正确识别的背景像素

---

## 🚀 使用指南

### 步骤1：验证修改
```bash
cd /home/neolux/workspace/2026/grad/code
cat mmsegmentation/configs/_base_/datasets/swrd.py | tail -10
# 应该看到新的 val_evaluator 配置
```

### 步骤2：训练或验证模型

**完整训练（含验证）：**
```bash
python mmsegmentation/tools/train.py \
    mmsegmentation/configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py \
    --work-dir work_dirs/segformer_new \
    --gpu-id 0
```

**仅验证已训练的模型：**
```bash
python mmsegmentation/tools/test.py \
    mmsegmentation/configs/swrd/segformer_mit-b0_4xb2-40k_swrd-512x512.py \
    work_dirs/segformer_mit-b0_4xb2-40k_swrd-512x512/latest.pth \
    --work-dir work_dirs/segformer_test
```

### 步骤3：检查 Scalar 日志

**位置1：实时日志输出**
```
stdout 中会显示：
per class results:
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
| Class | IoU   | Acc   | Dice  | mIoU  | mAcc  | mDice | ...  |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
| 0     | 45.2  | 67.8  | 54.3  | ...   | ...   | ...   | ...  |
| 1     | 38.5  | 62.1  | 49.2  | ...   | ...   | ...   | ...  |
| ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...  |
+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
```

**位置2：JSON 日志文件**
```
${work_dir}/vis_data/scalars.json
```

每行是一个JSON对象，包含所有指标：
```json
{
  "aAcc": 98.36,
  "mIoU": 35.65,
  "mAcc": 50.15,
  "mDice": 42.31,
  "mFscore": 38.42,
  "mPrecision": 48.22,
  "mRecall": 35.65,
  "step": 8000
}
```

### 步骤4：分析结果

**使用 Python 分析：**
```python
import json
import pandas as pd

# 读取日志
with open('work_dirs/segformer_new/vis_data/scalars.json', 'r') as f:
    logs = [json.loads(line) for line in f]

df = pd.DataFrame(logs)

# 查看关键指标趋势
print("=== 性能指标统计 ===")
print(df[['step', 'mIoU', 'aAcc', 'mDice', 'mFscore']].tail(10))

# 检查是否满足要求
final_metrics = logs[-1]
print(f"\n最终指标:")
print(f"✓ aAcc (像素准确率): {final_metrics['aAcc']:.2f}% {'✓' if final_metrics['aAcc'] > 75 else '❌'}")
print(f"✓ mIoU (平均IoU): {final_metrics['mIoU']:.2f}% {'✓' if final_metrics['mIoU'] > 80 else '❌'}")
```

---

## 📋 9类缺陷对应的 Scalar 指标位置

```
类别索引    缺陷名称         对应指标位置
0         气孔 (Porosity)    IoU[0], Acc[0], Dice[0], Precision[0], Recall[0]
1         夹渣 (Slag)        IoU[1], Acc[1], Dice[1], Precision[1], Recall[1]
2         裂纹 (Crack)       IoU[2], Acc[2], Dice[2], Precision[2], Recall[2]
3         咬边 (Undercut)    IoU[3], Acc[3], Dice[3], Precision[3], Recall[3]
4         未熔合 (Lack of F.) IoU[4], Acc[4], Dice[4], Precision[4], Recall[4]
5         未焊透 (Penetration) IoU[5], Acc[5], Dice[5], Precision[5], Recall[5]
6         缺陷6            IoU[6], Acc[6], Dice[6], Precision[6], Recall[6]
7         缺陷7            IoU[7], Acc[7], Dice[7], Precision[7], Recall[7]
8         背景 (Background)  IoU[8], Acc[8], Dice[8], Precision[8], Recall[8]
```

---

## 🎯 性能优化建议

如果 mIoU 未达到 0.8，可以尝试：

1. **增加训练轮数**
   ```python
   train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)
   ```

2. **调整学习率**
   ```python
   optim_wrapper = dict(
       type='OptimWrapper',
       optimizer=dict(type='AdamW', lr=0.0001, ...),  # 增加学习率
   )
   ```

3. **使用更强的数据增强**
   ```python
   # 在 train_pipeline 中添加
   dict(type='RandomRotate', prob=0.5, degree=10),
   dict(type='GaussBlur', sigma_range=(0.1, 2.0), prob=0.2),
   ```

4. **使用更好的预训练权重**
   ```python
   checkpoint = 'https://mmassets.onedl.ai/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66b8de61.pth'
   ```

---

## 📚 参考文档

- IoUMetric 源代码：`/mmsegmentation/mmseg/evaluation/metrics/iou_metric.py`
- 配置文件：`/mmsegmentation/configs/_base_/datasets/swrd.py`
- 训练脚本：`/mmsegmentation/tools/train.py`

