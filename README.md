# 虚拟酵母扰动响应预测 · GOAI 2026 赛道三（虚拟细胞方向）

> 基于条件响应预测的酿酒酵母虚拟细胞建模 —— 小样本、高维、零样本外推的蛋白组扰动响应预测

## 一句话概述

构建了一个**残差分解 + 上下文先验 + GO 通路注意力 + control/delta 低秩重构 + 菌株遗传特征 + 化学迁移融合**的混合模型体系，通过场景自适应映射（各测试场景选用验证集口径最强模型组合）实现跨菌株/跨化合物的蛋白质组扰动响应预测。**验证集四场景蛋白 R² 中位数 0.690~0.870（5-seed 集成均值），测试集真值自评加权总分 0.6349（全合规历史最高）**。

---

## 1. 任务与数据

### 1.1 科学问题

AI 虚拟细胞（AI Virtual Cell, AIVC）的目标：给定菌株、化合物、培养基、温度、时间等条件，预测细胞蛋白质组的整体重塑（5243 个蛋白的丰度向量）。核心挑战是**小样本高维零样本外推**——训练集不足 6000 样本、输出 4422 维（过滤后），且测试含训练集未见的化合物、菌株及双重未知组合。

### 1.2 数据

| 项 | 值 |
|---|---|
| 训练样本 | 5,920（处理 5,078 / 对照 751 / 质控 91） |
| 测试样本 | 4,454（处理 4,226 / 对照 202 / 质控 26） |
| 蛋白维度 | 5,243 → 过滤缺失率<80% 后 4,422 |
| 菌株 / 化合物 | 6 菌株（train 5 + val BAI）/ 56 扰动（含对照） |
| 条件 | 2 培养基（YNB+CSM glucose/galactose）× 2 温度（30/37）× 6 时间点（15~240min）× 2 质谱平台 |

### 1.3 评测

六个模块（官方口径）：M1 原始 FC（PCC of Δ, 25%）+ M2 绝对保真度（20%）+ M3 上下文残差（20%）+ M4 药物残差（20%）+ M5 双盲/时间（10%）+ M6 DEP（5%）。**65% 权重围绕 Δ = 处理 − 对照**，评测在四个 OOD 场景分别进行：新化合物（val_chem_only）、新菌株（val_strain_only）、双重未知（val_both）、时间外推（val_time）。

---

## 2. 方法

### 2.1 设计理念

- **结构对齐**：模型组件与评测维度对应——残差分解 B（基线）/S（共享应激）/C（化合物特异）/T（菌株调制）分别对准 M2/M3/M4；
- **上下文先验**：菌株×培养基×温度×时间分组均值作为蛋白空间先验（ctx_prior），为未见菌株提供"细胞状态锚点"；
- **外部知识**（8/11 统一开放榜）：化合物结构（Morgan 指纹 + RDKit 描述符）、蛋白序列（ESM2）、功能注释（GO 通路）、菌株遗传（1011 项目 SNP 距离）；
- **全合规**：所有统计量（缺失率/均值/标准化/SVD/迁移池）仅从 train 划分计算；测试真值仅用于最终自评（官方允许"可自评不可训练"）。

### 2.2 方法演进（三轮）

**阶段一 · 残差分解（v2.x）**：MLP 主干 + B/S/C/T 四分支残差分解 + 可见性门控 + 对照监督 + LOO 组件监督。v2.9 加入菌株×化合物交互项与自适应损失权重。最优为 3×v2.1 集成。

**阶段二 · 外部特征（v3.x）**：开放榜后引入 Morgan 指纹、ESM2 序列先验、STRING PPI、UniProt GO 通路注意力。**v3.7（GO 通路注意力）为最强单模型**：同一通路蛋白共享响应偏置，对 unseen 菌株场景提升 +0.013。

**阶段三 · 审计重构（v5.x，8/16 接手）**：
1. **P0 泄漏审计**（`_audit_leak.py`）：发现并修复 3 处统计量越界——chem_emb_init 曾用 val 真值（6 个 val-only 化合物初始��直接编码其真实平均响应）、matched control 曾跨划分（12 个训练样本对照来自 val）、迁移池曾含 val 对照。全部改为 train-only。
2. **control/delta 低秩重构（v5.0）**：按评审建议拆分"状态模型（预测匹配对照）+ 扰动模型（低秩 U@z_delta）"，实证其**单模型不如 v37 共享编码直接回归**（control 监督弱 + 低秩表达受限），但作为**集成成分有互补价值**。
3. **组件监督 + 结构特征（v5.1）**：补 LOO μ_ctx/μ_drug 残差监督 + 10 项 RDKit 描述符 → 单模型 strain +0.089/both +0.094。
4. **菌株遗传特征（v5.2）**：1011 项目 SNP 距离 → 菌株到训练菌株的遗传距离向量。**关键发现：test 新菌株 CRD 与训练菌株 CGD 遗传距离仅 0.383**——遗传近的菌株响应可迁移。单模型 strain +0.038/both +0.031。
5. **可靠性门控（v5.3）**：由相似度/支持数驱动的连续门控，实测**无收益**（与 v4.9 结论一致，门控非瓶颈），负面结论已记录。

### 2.3 最终方案

**场景自适应映射 + 自适应迁移融合**（各场景用验证集口径最强组合）：

| 场景 | 模型组合 | val 蛋白R²中位 |
|---|---|---|
| test_chem_only | 3×v2.1 + v35 | 0.870 |
| test_strain_only | **0.75×v37 + 0.25×v5.2** | **0.690** |
| test_both | **0.75×v37 + 0.25×v5.2** | **0.761** |
| test_time | 3×v2.1 + v35 | 0.833 |

**自适应化学迁移融合**：对未见化合物，用"同菌株条件对齐 + 指纹相似度 top-k 加权"从训练池迁移 Δ，融合权重 α = 0.1 + 0.2·clamp((sim−0.1)/0.4)（**α 形式由 val 伪新化合物敏感性扫描确定，未用 test 真值调参**）。高相似化合物（Tamoxifen↔4-OH-Tamoxifen 相似度 0.978）迁移信号强，低相似（FCCP 0.085）必须小 α 防噪声。

---

## 3. 结果

### 3.1 验证集（5-seed 伪测试均值±std）

| 场景 | 蛋白均值 | MC 基线 | v37 单 | v5.2 单 | **最终集成** |
|---|---:|---:|---:|---:|---:|
| val_chem_only | -0.037 | 0.632 | 0.864 | 0.833 | **0.869±0.000** |
| val_strain_only | -0.036 | 0.595 | 0.672 | 0.618 | **0.690±0.002** |
| val_both | -0.064 | 0.702 | 0.754 | 0.706 | **0.762±0.001** |
| val_time | -0.009 | 0.629 | 0.826 | 0.782 | **0.831±0.001** |

FC PCC：0.427 / 0.355 / 0.236 / 0.607（集成 5-seed 均值）。集成稳定性极佳（std ≤ 0.004，v37 主导）；单模型在 unseen 菌株场景有 seed 波动（std 0.027~0.033，小样本特性）。

### 3.2 测试集真值自评（最终提交 prediction_v52ens.csv，0.6349）

| 场景 | M1:FC(25%) | M2:绝对(20%) | M3:ctx残差(20%) | M4:drug残差(20%) | M5:双盲(10%) | M6:DEP(5%) | 总分 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| test_chem_only | 0.468 | 0.922 | 0.397 | 0.468 | — | 0.771 | **0.513** |
| test_strain_only | 0.827 | 0.769 | 0.827 | 0.826 | — | 0.903 | **0.736** |
| test_both | 0.556 | 0.808 | 0.556 | 0.556 | 0.605 | 0.848 | **0.626** |
| test_time | 0.624 | 0.917 | 0.561 | 0.562 | 0.779 | 0.860 | **0.685** |

**加权总分 = 0.6349**（历史提交对比：migfusion 0.6340 [含审计前特征，存档]、final_5243 0.6295）。

### 3.3 科学验证：通路富集

对预测 Δ 做 Mann-Whitney U 通路富集，预测高效应蛋白显著聚集于已知功能模块：

| 通路 | val_chem | val_strain | val_both | val_time |
|---|---|---|---|---|
| 氨基酸转运 | p<0.001 | p<0.001 | p<0.001 | p<0.001 |
| 半乳糖代谢 | p<0.05 | p<0.001 | p<0.001 | — |
| 氧化还原/ROS | — | p<0.01 | p<0.01 | — |
| 核糖体蛋白 | — | — | — | p<0.05 |

Top3 高效应蛋白：PMA2（质膜质子泵，|Δ|=1.72）、CWP1（细胞壁蛋白，1.50）、PLB2（磷脂酶，1.43）——均为文献记载的酵母胁迫响应蛋白。

---

## 4. 合规与外部数据披露

- **官方数据**：仅 train 划分用于训练与统计量估计；测试真值仅用于最终自评（官方允许"可自评不可训练"）
- **8/16 主动泄漏审计**：修复 3 处统计量越界（chem_emb_init 用 val 真值、matched control 跨划分、迁移池含 val 对照），全部改为 train-only；审计脚本 `_audit_leak.py` 开源
- **外部资源**（8/11 统一开放榜，全部披露）：
  - 化合物 Morgan 指纹 + RDKit 描述符：PubChem SMILES（train 43 + test 新 17），RDKit 2026.03.5
  - 蛋白序列 embedding：ESM2 `facebook/esm2_t6_8M_UR50D`（Meta），UniProt 酿酒酵母参考蛋白组（559292）
  - GO 通路注释：UniProt GO 生物过程（92 个高频通路）
  - STRING PPI v12（4932）
  - 菌株遗传距离：1011 酵母基因组项目 SNP 距离矩阵
- **复现性**：随机种子（1/2/42/43/44）硬编码、关键超参硬编码、全权重开源

---

## 5. 复现

### 环境

```
python>=3.10, torch>=2.12.0+cu126, numpy, pandas, scipy, scikit-learn, tqdm, rdkit, python-docx
```

### 从头训练

```bash
# 1. 特征管线（修复版 + 外部特征 + 描述符 + 遗传特征）
python code/01_data_prep.py
python code/02_features.py          # 泄漏修复版（chem_emb_init 仅 train）
python code/12_features_v30.py      # Morgan + ESM2
python code/15_features_go.py       # GO 通路矩阵
python code/17_features_ppi.py      # STRING PPI
python code/_fix_test_morgan.py     # test 新化合物指纹
python code/_features_desc.py       # RDKit 描述符
python code/_strain_genome_feats.py # 菌株 SNP 遗传距离

# 2. 训练最终管线模型
python code/05b_train_v21.py 42     # 3×v2.1（seed 42/43/44）
python code/05b_train_v21.py 43
python code/05b_train_v21.py 44
python code/05_train_v35.py 42      # v3.5
python code/05_train_v37.py 42      # v3.7 GO 通路注意力
python code/05_train_v50.py 42      # v5.2 control/delta + 遗传特征（seed 42/43/44）

# 3. 提交（场景自适应 + 自适应迁移融合）
python code/07n_submit_v50ens.py    # → prediction_v50ens_base.csv
python code/08c_mig_fusion_adaptive.py data/prediction_v50ens_base.csv data/prediction_v52ens.csv
```

### 直接推理 + 测试自评

```bash
python code/07n_submit_v50ens.py
python code/08c_mig_fusion_adaptive.py data/prediction_v50ens_base.csv data/prediction_v52ens.csv
python code/_test_score.py data/prediction_v52ens.csv --ctrl train   # 0.6349
```

---

## 6. 仓库结构

```
.
├── README.md                       # 本文件
├── 初赛方案文档.docx               # 初赛方案（完整叙事）
├── code/                           # 最终管线源码（31 个脚本）
│   ├── 01_data_prep.py ~ 09_submit_5243.py   # 主流程
│   ├── 04_model_v21/v37/v50.py     # 三个最终模型
│   ├── 05b/05_train_*.py           # 训练脚本
│   ├── 07j/07n_submit_*.py         # 场景自适应提交
│   ├── 08c_mig_fusion_adaptive.py  # 自适应迁移融合（最终）
│   ├── _features_desc.py / _strain_genome_feats.py / _reliability_feats.py
│   ├── _test_score.py / _pseudo_test.py / _audit_leak.py / _eval_*.py
│   └── build_doc.py                # 初赛文档生成器
├── data/                           # 关键数据（feats/元数据/外部特征）
├── models/                         # 最终方案必需权重（3×v2.1 + v35 + v37 + v5.2×5）
├── submission_file/                # 提交文件
│   ├── prediction_v52ens.csv       # ★ 最终：test 0.6349（全合规）
│   ├── prediction_migfusion.csv    # 历史存档：0.6340
│   └── prediction_final_5243.csv   # 历史存档：0.6295
└── docs/
    └── 交接文档.md                 # 完整工作记录（37 章，含审计与全步骤验证）
```

## 致谢

感谢 Datawhale 与 GOAI 2026 主办方提供赛题、数据与教程。
