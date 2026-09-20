# CNN 卷积层 CP 分解压缩实验

本目录是矩阵分析期末大作业的实验代码，目标是用 CP 分解压缩 CNN 卷积层，并研究不同微调策略对准确率恢复的影响。

核心结论：

- 直接 CP 替换卷积层几乎不可用，准确率接近随机猜测。
- 全局微调（Global FT）在 Fashion-MNIST 上恢复能力有限，精度损失明显。
- 迭代式逐层微调（Layer-wise FT）显著优于全局微调，优化更稳定。
- 数据感知逐层微调（Data-aware FT）在两个数据集和两个模型上均达到或略超 dense baseline，同时分别降低 51.44%（LeNet-5）和 77.50%（SmallCNN）FLOPs。

## 代码结构

```text
code/lenet_cp_mnist.py             主实验脚本：训练 baseline、CP 分解、微调、统计参数和 FLOPs
code/run_multi_seed_experiments.py 5-seed 主实验批量运行和 mean/standard error 汇总
code/environment.yml               Conda 环境配置
code/results/*.csv                 per-seed 和 multi-seed summary 结果
```

生成文件：

```text
data/                              MNIST/Fashion-MNIST 数据
code/checkpoints/                  baseline 和 CP 模型 checkpoint
```

数据和 checkpoint 都是生成产物，已被 `.gitignore` 忽略。

## 环境与快速检查

创建环境：

```bash
conda env create -f code/environment.yml
conda activate matrix-cp
```

快速检查 LeNet-5 rank 8 的压缩统计：

```bash
python code/lenet_cp_mnist.py --smoke
```

预期输出：

```text
Baseline FLOPs: 852.1560 KFLOPs
CP FLOPs:       413.8200 KFLOPs
Conv params:    2572 -> 414
```

检查 SmallCNN rank 16：

```bash
python code/lenet_cp_mnist.py --model smallcnn --smoke --rank 16
```

预期压缩统计：

```text
Baseline:  426,122 params, 23,296 conv params, 15,619.9680 KFLOPs
CP rank16: 405,802 params,  2,976 conv params,  3,515.0080 KFLOPs
Reduction: 7.83x conv parameter ratio, 77.50% fewer FLOPs
```

## 整体算法流程

```text
1. 训练 dense CNN baseline
2. 对目标卷积层权重做 CP-ALS 分解
3. 将一个 Conv2d 替换为 4 个小卷积层（1x1 + 深度竖直 + 深度水平 + 1x1）
4. 统计参数量和 FLOPs
5. 对 CP 模型做不同策略的微调
6. 比较准确率恢复效果
```

最有效的流程（Data-aware FT）：

```text
dense baseline
-> 按深度逆序逐层 CP 替换
-> 每替换一层，先做 data-aware activation reconstruction（最小化该层输出 MSE）
-> 每替换一层，再做分类微调
-> 最后做全局分类微调
```

## 数据集

| 数据集 | 输入 | 类别数 |
|---|---:|---:|
| MNIST | `1 x 28 x 28` | 10 |
| Fashion-MNIST | `1 x 28 x 28` | 10 |

归一化设置：

| 数据集 | mean | std |
|---|---:|---:|
| MNIST | 0.1307 | 0.3081 |
| Fashion-MNIST | 0.2860 | 0.3530 |

## 模型

### LeNet-5

```text
conv1: 1 -> 6,  kernel 5x5, padding 2
conv2: 6 -> 16, kernel 5x5
fc1(400->120), fc2(120->84), fc3(84->10)

CP 设置：rank = 8，目标层 = conv1, conv2
```

### SmallCNN

```text
conv1: 1  -> 16, kernel 3x3
conv2: 16 -> 32, kernel 3x3
conv3: 32 -> 64, kernel 3x3
fc1(3136->128), fc2(128->10)

CP 设置：rank = 16，目标层 = conv1, conv2, conv3
```

## 训练设置

所有主要结果使用 5 个随机种子汇报（standard error = sample std / sqrt(5)）：

```text
seeds          = 0, 1, 2, 3, 4
batch_size     = 64
baseline lr    = 1e-3，训练 10 epochs，Adam 优化器
CP-ALS 迭代次数 = 50
```

微调超参数：

| 方法 | epochs | lr |
|---|---|---|
| Global FT | 5 | `1e-4` |
| Layer-wise FT（每层） | 2 | `1e-3` |
| Layer-wise FT（最终全局） | 5 | `1e-4` |
| Data-aware 激活重构（每层） | 1 | `1e-3` |

## CP-ALS 算法

卷积核权重是一个四阶张量 $W \in \mathbb{R}^{O \times I \times H \times W}$，CP 分解将其近似为：

```text
W[o,i,h,w] ~= sum_{r=1}^R lambda_r * A[o,r] * B[i,r] * C[h,r] * D[w,r]
```

ALS 每次固定其余三个 factor，只更新一个 factor：

```text
更新 factor n：U^(n) <- X_(n) @ K^(n) @ pinv(K^(n).T @ K^(n))
其中 K^(n) 为其余 factor 的 Khatri-Rao 积，X_(n) 为张量沿第 n 维的矩阵展开
```

每次更新后归一化 factor 列，尺度吸收进 `lambda`（`weights`）。

## CP 卷积层替换方式

原始一个 Conv2d 替换为 4 个小卷积层：

```text
1x1 pointwise conv:        in_channels -> rank
depthwise vertical conv:   rank -> rank, kernel_h x 1, groups=rank
depthwise horizontal conv: rank -> rank, 1 x kernel_w, groups=rank
1x1 pointwise conv:        rank -> out_channels（含偏置）
```

精确参数量为 $R(O + I + H + W) + O$（最后一层含偏置 $O$ 项）。

## 已实现方法

### 1. CP 替换，无微调（cp_no_ft）

直接 CP-ALS 替换目标卷积层，不做训练。准确率接近随机猜测（约 10%）。

### 2. 全局微调（global_ft）

一次性替换所有目标卷积层，然后对整个 CP 模型做分类微调：

```text
loss = CrossEntropy(logits, label)
epochs = 5, lr = 1e-4
```

### 3. 迭代式逐层微调（layerwise_ft）

按深度逆序逐层替换卷积层，每替换一层就微调整个模型：

```text
for layer in reversed(conv_layers):
    replace layer with CP structure
    finetune whole model for 2 epochs (lr=1e-3)
final global finetune for 5 epochs (lr=1e-4)
```

### 4. 数据感知逐层微调（data_aware_layerwise_ft）

在逐层微调基础上，每层替换后先做激活重构：

```text
for layer in reversed(conv_layers):
    save dense reference layer
    replace layer with CP structure
    minimize MSE(CP_output(x_l), dense_output(x_l)) for 1 epoch (lr=1e-3)
    finetune whole model for 2 epochs (lr=1e-3)
final global finetune for 5 epochs (lr=1e-4)
```

其中 `x_l` 是当前（已部分替换的）模型前向传播到该层时的真实输入特征图。

## 主要实验结果

所有结果：5 seed 均值 ± 标准误差（%）。

### MNIST

| 模型 | Rank | Baseline | CP 无微调 | Global FT | Layer-wise FT | Data-aware FT |
|---|---:|---:|---:|---:|---:|---:|
| LeNet-5 | 8 | 98.89 ± 0.06 | 9.21 ± 0.84 | 96.95 ± 0.24 | 98.77 ± 0.07 | **99.02 ± 0.02** |
| SmallCNN | 16 | 99.08 ± 0.06 | 10.79 ± 0.62 | 97.96 ± 0.14 | 99.18 ± 0.03 | **99.33 ± 0.02** |

### Fashion-MNIST

| 模型 | Rank | Baseline | CP 无微调 | Global FT | Layer-wise FT | Data-aware FT |
|---|---:|---:|---:|---:|---:|---:|
| LeNet-5 | 8 | 90.17 ± 0.14 | 9.59 ± 0.89 | 83.89 ± 0.71 | 89.39 ± 0.11 | **90.31 ± 0.08** |
| SmallCNN | 16 | 92.29 ± 0.07 | 9.49 ± 0.28 | 87.06 ± 0.45 | 92.03 ± 0.04 | **92.43 ± 0.09** |

### 压缩统计

| 模型 | Rank | 卷积参数 | 压缩比 | Baseline FLOPs | CP FLOPs | FLOPs 降低 |
|---|---:|---:|---:|---:|---:|---:|
| LeNet-5 | 8 | 2,572 → 414 | 6.21× | 852.16 KFLOPs | 413.82 KFLOPs | 51.44% |
| SmallCNN | 16 | 23,296 → 2,976 | 7.83× | 15,619.97 KFLOPs | 3,515.01 KFLOPs | 77.50% |

完整 per-seed 结果和汇总分别保存在：

```text
code/results/multi_seed_results_seeded.csv
code/results/multi_seed_summary_seeded.csv
```

## 复现命令

### 5-seed 主实验

```bash
python code/run_multi_seed_experiments.py \
  --seeds 0,1,2,3,4 \
  --output code/results/multi_seed_results_seeded.csv \
  --summary-output code/results/multi_seed_summary_seeded.csv
```

脚本会自动跳过已存在的 `(dataset, model, seed, method)` 组合，中断后可直接续接。

### 单次实验示例（LeNet-5，Fashion-MNIST，data-aware FT）

```bash
python code/lenet_cp_mnist.py \
  --dataset fashion-mnist \
  --model lenet5 \
  --epochs 10 \
  --rank 8 \
  --seed 0 \
  --layerwise-finetune \
  --layerwise-reconstruction-epochs 1 \
  --layerwise-reconstruction-lr 1e-3 \
  --layerwise-checkpoint code/checkpoints/lenet5_fmnist_cp_data_aware_seed0.pt
```
