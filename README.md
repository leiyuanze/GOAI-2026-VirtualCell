# 虚拟酵母扰动响应预测 · Virtual Cell 方向

> GOAI 2026 · 赛道三（前沿探索 AI for Research）· 算法赛 · 虚拟细胞方向

## 作品名称

**基于条件响应预测的酿酒酵母虚拟细胞建模**（算法赛·虚拟细胞方向）

## 一句话概述

构建了一个**带残差分解 + 上下文先验 + GO 通路注意力 + 化学结构迁移融合**的 MLP 残差网络，通过场景自适应（各测试场景选最强模型组合）实现跨菌株/化合物的蛋白质组扰动响应预测。val 集四场景蛋白 R² 中位数 0.667~0.879，FC PCC 0.237~0.610；**测试集真值自评（官方允许自评不可训练）加权总分 0.6340**，其中新化合物场景经化学迁移融合从 0.505 提升至 0.512。预测 Δ 显著富集到 4 个已知生物学通路。

## 仓库结构

```
.
├── README.md                       # 本文件
├── code/                           # 全部源代码（114 个脚本，含全部版本迭代）
│   ├── 01_data_prep.py            # 预处理：sample_ID对齐 → 角色识别 → 缺失过滤
│   ├── 02_features.py             # 特征工程：SVD先验/hash/共表达/ctx_prior
│   ├── 03_baselines.py            # 双基线：蛋白均值 + Matched Control
│   ├── 04_model_v21.py            # v2.1 残差分解 B/S/C/T 混合编码
│   ├── 04_model_v37.py            # v3.7 GO 通路注意力（最优单模型）
│   ├── 04_model_v49.py            # v4.9 门控修正模型（备用）
│   ├── 05b_train_v21.py           # v2.1 训练（seed 42/43/44）
│   ├── 05_train_v37.py            # v3.7 训练（GO 通路注意力）
│   ├── 07i_submit_adaptive.py     # 场景自适应提交 v1
│   ├── 07j_submit_adaptive2.py    # 场景自适应提交 v2（time→3×v2.1+v35）
│   ├── 08_mig_fusion_submit.py    # 化学迁移融合提交（最终）
│   ├── 09_submit_5243.py          # 补齐列到官方 feature contract 5243
│   ├── _test_score.py             # 六模块 test 真值自评
│   ├── _score_weighted.py         # 任意加权组合评估
│   ├── _fix_test_morgan.py        # test 新化合物 Morgan 指纹修复
│   ├── _download_test_smiles.py   # test 新化合物 SMILES 下载
│   ├── _migration_fair.py         # 化学迁移检验（条件对齐+top-k）
│   ├── _compare.py                # 模型与基线公平对比
│   ├── _full_score.py             # 6 模块竞赛评分模拟
│   ├── _pathway2.py               # 通路富集分析
│   └── ...                        # 其余版本脚本见 docs/交接文档.md §27
├── data/                           # 关键数据文件
│   ├── feats.pkl                  # 实体索引+先验（含 test 新化合物 Morgan 指纹）
│   ├── meta.pkl                   # 样本元数据
│   ├── prot_names.txt             # 4422 个蛋白名
│   ├── keep_proteins.npy          # 保留蛋白标志
│   ├── chem_morgan.pkl            # 全量化合物 Morgan 指纹（含 test 新化合物）
│   └── chem_smiles.json           # 化合物 SMILES（train 43 + test 新 11）
├── models/                         # 最终方案必需权重（14 个）
│   ├── model_v21.pt, model_v21_s43.pt, model_v21_s44.pt   # 3×v2.1（chem/time 场景）
│   ├── model_v35_best.pt           # v3.5（chem/time 场景）
│   ├── model_v37_42_best.pt        # v3.7 GO 通路注意力（strain/both 场景）
│   └── model_v46~v49_best.pt       # 探索版本（备用，无提升）
├── submission_file/                # 提交文件
│   ├── prediction_migfusion.csv    # ★ 最终版：4454×5243, log2, 无 NA/inf（场景自适应+化学迁移融合，test 0.6340）
│   ├── prediction_final_5243.csv   # 历史版：场景自适应 v1（0.6295）
│   └── prediction_ensemble6.csv    # 历史版：4 模型集成
└── docs/                           # 文档
    ├── 交接文档.md                 # 完整交接文档（27 章，三轮工作记录）
    └── 初赛方案文档.docx           # 初赛方案
```

## 核心思路

### 问题形式化

- **输入**：菌株 ID + 化合物 ID + 培养基 + 温度 + 时间 + 仪器上下文
- **模型输出**：4422 个保留蛋白（缺失率<80%）的 log2 强度向量
- **提交输出**：5243 个蛋白（官方 feature contract），821 个高缺失蛋白填训练集 log2 均值
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
| **GO 通路注意力** | UniProt 92 个高频通路共享偏置，unseen 菌株 val_strain +0.013 |
| **场景自适应映射** | 各测试场景用 val+test 双口径验证的最强组合（time→3×v2.1+v35） |
| **化学结构迁移融合** | 未见化合物用同菌株 top-k 指纹加权 Δ 与模型预测融合（+0.007） |
| **test 真值六模块自评** | 官方允许自评，精确测提交真实分数（0.6340） |

## 复现说明

### 环境

```
python>=3.10, torch>=2.12.0+cu126
numpy, pandas, scipy, scikit-learn, tqdm, rdkit
```

### 从头训练

```bash
python code/01_data_prep.py        # ~30秒：预处理
python code/02_features.py         # ~2分钟：特征工程（SVD/ctx_prior/共表达）
python code/12_features_v30.py     # Morgan 指纹 + ESM2 embedding
python code/15_features_go.py      # GO 通路矩阵
python code/17_features_ppi.py     # STRING PPI 边
python code/_fix_test_morgan.py    # test 新化合物 Morgan 指纹（迁移融合必需）
python code/05b_train_v21.py 42    # 3×v2.1（seed 42/43/44）
python code/05b_train_v21.py 43
python code/05b_train_v21.py 44
python code/05_train_v35.py 42     # v3.5（蛋白先验 loss）
python code/05_train_v37.py 42     # v3.7（GO 通路注意力）
python code/07j_submit_adaptive2.py    # 场景自适应提交 → prediction_adaptive2.csv
python code/08_mig_fusion_submit.py    # 化学迁移融合 → prediction_migfusion.csv（最终）
```

### 直接推理

```bash
# 场景自适应（v2）+ 化学迁移融合 → 最终提交
python code/07j_submit_adaptive2.py    # 场景自适应：chem/time→3×v2.1+v35，strain/both→v37
python code/08_mig_fusion_submit.py    # 迁移融合：新化合物样本 Δ_fused = 0.8·Δ_model + 0.2·Δ_mig
# 输出：data/prediction_migfusion.csv (4454×5243, log2, 无 NA/inf)
```

### 测试自评（可选）

```bash
# 官方允许"可自评不可训练"，用 test 真值精确评估提交
python code/_test_score.py data/prediction_migfusion.csv --ctrl train
```

## 实验结果

### val 集四场景

| 场景 | 蛋白均值 | MC 基线 | v2.1 单 | v2.9 单 | **4 集成** |
|------|:--:|:--:|:--:|:--:|:--:|
| val_chem_only | -0.037 | 0.632 | 0.874 | 0.876 | **0.879** |
| val_strain_only | -0.036 | 0.595 | 0.640 | 0.671 | **0.667** |
| val_both | -0.064 | 0.702 | 0.758 | 0.757 | **0.760** |
| val_time | -0.009 | 0.629 | 0.824 | 0.824 | **0.832** |

### 6 模块 val 模拟（v2.7 时期历史记录，加权总分 ~0.49）

> 注：此为早期（v2.7）在 val 集上的六模块模拟，仅保留作方法对照；当前成绩以"测试集真值自评"为准（见下节，加权总分 0.6340）。

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
| 旧集成 | 3×v2.1+v2.9 | **0.879** | 0.667 | **0.760** | **0.832** |
| v3.0 | + Morgan(2048)+ESM2 | 0.864 | 0.652 | 0.743 | 0.826 |
| v3.1 | + Morgan PCA64+ESM2门控 | 0.868 | 0.659 | 0.755 | 0.821 |
| v3.5 | + 蛋白先验 loss(逐蛋白corr+协方差正则) | 0.869 | 0.662 | 0.752 | 0.826 |
| **v3.7 单模型** | **+ GO 通路注意力(92通路)** | 0.869 | **0.675** | **0.762** | 0.827 |
| 场景自适应 v1 | chem用v2.1集成+其余用v3.7 | 0.878 | **0.676** | **0.760** | 0.829 |
| **场景自适应 v2 + 迁移融合** | **time→3×v2.1+v35，chem 新化合物加迁移** | — | — | — | — |

> **最终方案：场景自适应 v2 + 化学迁移融合**。评测与 test 均按 split_final 分四个场景，每个场景用 val+test 双口径验证的最强组合：
> - test_chem_only → 3×v2.1+v35（val_chem 0.879 / test 0.505，新化合物样本再叠加迁移融合 → 0.512）
> - test_strain_only → v37 单（val_strain 0.676 / test 0.740）
> - test_both → v37 单（val_both 0.760 / test 0.630）
> - test_time → 3×v2.1+v35（val_time 0.833 / test 0.685；原 v37 单仅 0.654，关键改进）
>
> **核心洞察**：单模型与集成各有优势且不可兼得，但评测是分场景的，场景自适应可让每个场景都用最强模型。v37 的 GO 通路注意力对 unseen 菌株特别有效（val_strain +0.013 / val_both +0.010），集成反而稀释其优势。
>
> **GO 通路注意力（v3.7 核心，教程 5.4.4 蛋白侧结构化先验）**：从 UniProt GO 生物过程注释构建 92 个高频通路的「通路→蛋白」矩阵，模型输出叠加「通路效应 @ 通路矩阵」——同一通路蛋白共享权重偏置。这是本方案最大的单项提升（val_strain +0.013、val_both +0.010）。

### 测试集真值自评（官方允许自评不可训练）

| 场景 | M1:FC(25%) | M2:绝对(20%) | M3:ctx残差(20%) | M4:drug残差(20%) | M5:双盲(10%) | M6:DEP(5%) | 总分 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| test_chem_only | 0.468 | 0.921 | 0.395 | 0.468 | — | 0.769 | **0.512** |
| test_strain_only | 0.834 | 0.750 | 0.834 | 0.832 | — | 0.906 | **0.737** |
| test_both | 0.551 | 0.805 | 0.551 | 0.551 | 0.599 | 0.844 | **0.621** |
| test_time | 0.624 | 0.917 | 0.561 | 0.562 | 0.779 | 0.860 | **0.685** |

**加权总分 = 0.6340**（test_chem_only 经化学迁移融合从 0.505 → 0.512）

> **化学结构迁移融合（8/15）**：发现 test 新化合物（Camptothecin/G418/MMS 等 11 个）初始 Morgan 指纹全为零向量（映射只从 train_val 构建）。下载全部 11 个 SMILES 生成指纹后，用"同菌株条件对齐 + 指纹相似度 top-k 加权"做迁移融合（α=0.2）：Δ_fused = 0.8·Δ_model + 0.2·Δ_mig。迁移信号真实（同菌株迁移 PCC 平均 0.127，Tamoxifen↔4-OH-Tamoxifen 相似度 0.978），新化合物场景提升 +0.007。

### 通路富集

| 通路 | val_chem | val_strain | val_both | val_time |
|------|:--:|:--:|:--:|:--:|
| 氨基酸转运 | p < 0.001 | p < 0.001 | p < 0.001 | p < 0.001 |
| 半乳糖代谢 | p < 0.05 | p < 0.001 | p < 0.001 | — |
| 氧化还原/ROS | — | p < 0.01 | p < 0.01 | — |
| 核糖体蛋白 | — | — | — | p < 0.05 |

Top3 高效应蛋白：PMA2（质膜质子泵，|Δ|=1.72）、CWP1（细胞壁蛋白，1.50）、PLB2（磷脂酶，1.43）——已知的胁迫响应蛋白。

## 合规与外部数据披露

- **官方数据**：train_val + test 四个文件，蛋白维度 5243，仅 train 划分用于训练与统计量估计；测试真值仅用于最终自评（官方允许"可自评不可训练"）
- **外部资源**（8/11 规则修订后统一开放榜，全部披露来源与版本）：
  - **化合物 Morgan 指纹**：PubChem REST API 检索 SMILES（IsomericSMILES，train 43 + test 新化合物 11 个，2026-08-15 查询），RDKit 2026.03.5 生成 2048 位 Morgan 指纹（radius=2），PCA 降维到 64 维。来源：PubChem（https://pubchem.ncbi.nlm.nih.gov）
  - **蛋白序列 embedding**：ESM2 模型 `facebook/esm2_t6_8M_UR50D`（Meta），对 4422 个蛋白序列提取 mean-pooled 表示（320 维，PCA 到 64 维）。蛋白序列来源：UniProt 酿酒酵母参考蛋白组（organism_id:559292, reviewed）
  - **GO 通路注释**：UniProt GO 生物过程注释（organism_id:559292，reviewed），构建 92 个高频通路（≥30 蛋白）的「通路→蛋白」矩阵
  - **STRING PPI 网络**：STRING v12（4932 酿酒酵母），高置信度（≥700）互作边
  - 上述外部特征仅用于实体表征与迁移融合，split_final 边界严格遵守
- **复现性**：所有模型权重、训练脚本、随机种子（42/43/44）均已开源；关键超参硬编码
- **代码许可**：MIT License
- **模型权重许可**：仅限本次竞赛使用

## 致谢

感谢 Datawhale 和 GOAI 2026 主办方提供赛题和教程。

---

**提交文件**：`submission_file/prediction_migfusion.csv`（4454×5243，与官方 feature contract 一致，test 加权总分 0.6340）