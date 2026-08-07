# 虚拟酵母扰动响应预测 · Virtual Cell 方向

> GOAI 2026 · 赛道三（前沿探索 AI for Research）· 算法赛 · 虚拟细胞方向

## 作品名称

**基于条件响应预测的酿酒酵母虚拟细胞建模**（算法赛·虚拟细胞方向）

## 一句话概述

构建了一个**带残差分解 + 上下文先验 + 菌株×化合物交互项**的 MLP 残差网络，通过 4 模型混合集成实现跨菌株/化合物的蛋白质组扰动响应预测。在封闭数据榜 val 集四场景（chem_only/strain_only/both/time）蛋白 R² 中位数 0.667~0.879，FC PCC 0.237~0.610，且预测 Δ 显著富集到 4 个已知生物学通路。

## 仓库结构

```
.
├── README.md                       # 本文件
├── code/                           # 全部源代码
│   ├── 01_data_prep.py            # 预处理：sample_ID对齐 → 角色识别 → 缺失过滤
│   ├── 02_features.py             # 特征工程：SVD先验/hash/共表达/ctx_prior
│   ├── 03_baselines.py            # 双基线：蛋白均值 + Matched Control
│   ├── 04_model_v21.py            # v2.1 残差分解 B/S/C/T 混合编码
│   ├── 04_model_v25.py            # v2.5 输入级分离
│   ├── 04_model_v27.py            # v2.7 全 ctx_prior 注入 + unseen 均值 emb
│   ├── 04_model_v29.py            # v2.9 + 交互项 + 自适应 loss
│   ├── 05b_train_v21.py           # v2.1 训练（seed 42/43/44）
│   ├── 05q_train_v29.py           # v2.9 训练
│   ├── 07e_submit_v29.py          # 最终提交生成（4 模型集成）
│   ├── _compare.py                # 模型与基线公平对比
│   ├── _full_score.py             # 6 模块竞赛评分模拟
│   └── _pathway2.py               # 通路富集分析
├── data/                           # 关键数据文件
│   ├── feats.pkl                  # 实体索引+先验（154MB）
│   ├── meta.pkl                   # 样本元数据
│   ├── prot_names.txt             # 4422 个蛋白名
│   └── keep_proteins.npy          # 保留蛋白标志
├── models/                         # 4 个最终模型权重（18MB）
│   ├── model_v21.pt, model_v21_s43.pt, model_v21_s44.pt
│   └── model_v29_best.pt
├── submission_file/                # 最终提交
│   └── prediction_ensemble6.csv   # 4454×4422, log2, 无 NA/inf
```

## 核心思路

### 问题形式化

- **输入**：菌株 ID + 化合物 ID + 培养基 + 温度 + 时间 + 仪器上下文
- **输出**：4422 个蛋白的 log2 强度向量
- **评测**：6 个模块在 4 个 OOD 场景（M1 25% + M2 20% + M3 20% + M4 20% + M5 10% + M6 5%）

### 关键设计决策

| 设计点 | 原因 |
|------|------|
| **残差分解 B/S/C/T** | 65% 分数围绕 Δ=处理−对照，分解架构与评测模块 1-5 一一对应 |
| **输入级分离** | 修复 v2.1 的"伪解耦"，C/T 分支各吃独立特征 |
| **可见性门控** | T 分支 g_s = sigmoid × strain_seen（unseen 菌株 T 贡献归零） |
| **ctx_prior 注入** | 蛋白空间先验（4422 维→64 维）喂入混合编码器+T 分支 |
| **Unseen 均值 embedding** | 训练菌株 embedding 的均值作为未见兜底 |
| **菌株×化合物交互项** | bilinear MLP 捕捉双盲场景非加性效应（对应 M5） |
| **自适应 loss 权重** | Kendall uncertainty 自动平衡 4 个 loss |
| **批次校准分支** | calib head 只吃仪器/板号，与生物编码解耦 |
| **3×v2.1+v2.9 混合集成** | 不同架构+不同 seed 多模型平均 |

## 复现说明

### 环境

```
python>=3.10, torch>=2.12.0+cu126
numpy, pandas, scipy, scikit-learn, tqdm
```

### 从头训练

```bash
python code/01_data_prep.py     # ~30秒
python code/02_features.py      # ~2分钟
python code/05b_train_v21.py 42 # ~5分钟
python code/05b_train_v21.py 43 # ~5分钟
python code/05b_train_v21.py 44 # ~5分钟
python code/05q_train_v29.py 42 # ~10分钟
python code/07e_submit_v29.py   # 生成 prediction_ensemble6.csv
```

### 直接推理

```bash
python code/07e_submit_v29.py
# 输出：data/prediction_ensemble6.csv
```

## 实验结果

### val 集四场景

| 场景 | 蛋白均值 | MC 基线 | v2.1 单 | v2.9 单 | **4 集成** |
|------|:--:|:--:|:--:|:--:|:--:|
| val_chem_only | -0.037 | 0.632 | 0.874 | 0.876 | **0.879** |
| val_strain_only | -0.036 | 0.595 | 0.640 | 0.671 | **0.667** |
| val_both | -0.064 | 0.702 | 0.758 | 0.757 | **0.760** |
| val_time | -0.009 | 0.629 | 0.824 | 0.824 | **0.832** |

### 6 模块 val 模拟（加权总分 ~0.49）

| 模块 | val_chem | val_strain | val_both | val_time |
|------|:--:|:--:|:--:|:--:|
| M1 原始FC (25%) | 0.457 | 0.350 | 0.237 | 0.610 |
| M2 绝对保真度 (20%) | 0.933 | 0.816 | 0.862 | 0.909 |
| M3 上下文残差 (20%) | 0.430 | 0.346 | 0.233 | 0.545 |
| M4 药物残差 (20%) | 0.457 | 0.277 | 0.233 | 0.530 |
| M5 双盲/时间 (10%) | — | — | 0.496 | 0.763 |
| M6 DEP (5%) | 0.777 | 0.766 | 0.666 | 0.843 |
| **加权总分** | **0.517** | **0.413** | **0.407** | **0.667** |

### 迭代消融

| 版本 | 关键改动 | val_chem | val_strain | val_both | val_time |
|------|------|:--:|:--:|:--:|:--:|
| v2.0 | 基础残差分解 | 0.665 | 0.450 | 0.520 | 0.620 |
| **v2.1** | + 对照监督+组件 LOO | **0.874** | 0.640 | **0.758** | 0.824 |
| v2.5 | 输入级分离 | 0.876 | 0.607 | 0.676 | 0.833 |
| v2.7 | 全 ctx_prior+均值 emb | 0.877 | 0.650 | 0.737 | 0.830 |
| **v2.9** | + 交互项+自适应 loss | **0.876** | **0.671** | **0.757** | **0.824** |
| 4 集成 | 3×v2.1+v2.9 | **0.879** | 0.667 | **0.760** | **0.832** |

### 通路富集

| 通路 | val_chem | val_strain | val_both | val_time |
|------|:--:|:--:|:--:|:--:|
| 氨基酸转运 | p < 0.001 | p < 0.001 | p < 0.001 | p < 0.001 |
| 半乳糖代谢 | p < 0.05 | p < 0.001 | p < 0.001 | — |
| 氧化还原/ROS | — | p < 0.01 | p < 0.01 | — |
| 核糖体蛋白 | — | — | — | p < 0.05 |

Top3 高效应蛋白：PMA2（质膜质子泵，|Δ|=1.72）、CWP1（细胞壁蛋白，1.50）、PLB2（磷脂酶，1.43）——已知的胁迫响应蛋白。

## 合规

- **数据**：仅使用组委会发布的 train_val 数据 + 官方字段字典
- **无外部知识**：未引入 ESM/DNABERT/ChemBERTa 等预训练模型，未引入 STRING/GO/KEGG/PubChem 等外部数据库
- **复现性**：所有模型权重、训练脚本、随机种子均已开源
- **代码许可**：MIT License
- **模型权重许可**：仅限本次竞赛使用

## 致谢

感谢 Datawhale 和 GOAI 2026 主办方提供赛题和教程。

---


**提交文件**：`submission_file/prediction_ensemble6.csv`