# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 04 模型定义（v5.0：control + delta 低秩重构）
按 gpt1/gpt2 指示的核心重构：
  ŷ = ŷ_ctrl + Δ̂,  Δ̂ = U @ z_delta
- control 分支（状态模型）：专攻"该菌株在条件下的基线状态"，输入不含化合物
  - 监督：对照样本真值 + 处理样本的 matched control（train-only）
- response 分支（低秩扰动模型）：专攻"化合物引发的蛋白组扰动"
  - U ∈ R^(P×d) 由训练集处理样本 Δ 的 SVD 学习（蛋白共变基）
  - z_delta 只输出 d=64 维，大幅降低 4422 维直接回归的过拟合风险
- 保留 v37 已验证有效的：GO 通路注意力 + ESM2 序列先验（门控）+ 可见性门控
"""
import numpy as np
import torch
import torch.nn as nn


class VCellModel(nn.Module):
    def __init__(self, feats, P=4422, hidden=256, latent=64, d_lowrank=64,
                 response_basis=None, esm_dim=64, gate_mode='hard'):
        super().__init__()
        self.P = P
        self.latent = latent
        self.d_lowrank = d_lowrank
        self.gate_mode = gate_mode  # 'hard'（v5.0~v5.2 兼容）/ 'rel'（步骤12 可靠性门控）

        # ---- 共享 embedding ----
        self.strain_emb = nn.Embedding(feats['n_strains'], 16)
        self.chem_emb = nn.Embedding(feats['n_chems'], 32)
        self.sm_emb = nn.Embedding(feats['n_sm'], 4)
        self.ct_emb = nn.Embedding(feats['n_ct'], 8)
        self.src_emb = nn.Embedding(feats['n_src'], 4)
        self.ins_emb = nn.Embedding(feats['n_ins'], 4)
        self.plt_emb = nn.Embedding(feats['n_plt'], 16)
        with torch.no_grad():
            si = torch.tensor(feats['strain_emb_init'][:, :4], dtype=torch.float32)
            ci = torch.tensor(feats['chem_emb_init'], dtype=torch.float32)
            self.strain_emb.weight[:si.shape[0], :4].copy_(si)
            self.chem_emb.weight[:ci.shape[0], :ci.shape[1]].copy_(ci)

        # ---- ctx_prior 投影（蛋白空间先验）----
        self.ctx_prior_proj = nn.Sequential(
            nn.Linear(P, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64),
        )
        # Morgan 指纹投影（64 维输入）
        self.morgan_proj = nn.Sequential(
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
        )
        # ★ RDKit descriptors 投影（gpt2 步骤9：10 维描述符 → 16 维）
        self.desc_proj = nn.Linear(10, 16)
        # ★ 菌株遗传距离投影（gpt2 步骤11：SNP 距离到训练菌株 4 维 → 16 维）
        self.strain_genome_proj = nn.Linear(4, 16)

        # ---- Control 分支（状态模型）：strain + genome + medium + temp + time + ctx_prior + calib ----
        d_ctrl = 16 + 16 + 2 + 1 + 3 + 4 + 64 + (4 + 4 + 16)  # strain, genome, med, temp, time, sm, ctx_prior, calib
        self.ctrl_enc = nn.Sequential(
            nn.Linear(d_ctrl, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.GELU(),
        )
        self.fc_ctrl = nn.Linear(latent, P)

        # ---- Response 分支（低秩扰动）：strain + genome + chem + Morgan + desc + med + temp + time + ct + ctx_prior ----
        d_resp = 16 + 16 + 32 + 64 + 16 + 2 + 1 + 3 + 8 + 64
        self.resp_enc = nn.Sequential(
            nn.Linear(d_resp, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, d_lowrank), nn.LayerNorm(d_lowrank), nn.GELU(),
        )

        # ---- 低秩响应基 U（P × d_lowrank，从训练集 Δ SVD 学习，buffer 冻结）----
        if response_basis is not None:
            self.register_buffer('U', torch.tensor(response_basis, dtype=torch.float32))
        else:
            self.register_buffer('U', torch.zeros(P, d_lowrank))
        # 门控：seen 化合物/菌株控制响应强度
        self.gate_c = nn.Parameter(torch.tensor(3.0))
        self.gate_s = nn.Parameter(torch.tensor(3.0))

        # ---- 蛋白级残差头（小权重，增强绝对保真）----
        self.resid_head = nn.Linear(d_lowrank, P)
        self.resid_scale = nn.Parameter(torch.tensor(0.2))

        # ---- ESM2 序列先验 + GO 通路注意力（保留 v37 已验证组件，加在 Δ̂ 上）----
        self.register_buffer('esm2_emb', torch.tensor(feats['esm2_emb'], dtype=torch.float32))  # P × esm_dim
        self.w_esm = nn.Linear(d_lowrank, esm_dim)
        self.gate_esm = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('go_mat', torch.tensor(feats['go_mat'], dtype=torch.float32))  # K × P
        self.go_effect = nn.Linear(d_lowrank, feats['n_go'])
        self.gate_go = nn.Parameter(torch.tensor(0.1))

        # ---- 基线 bias（gmean）----
        self.ctrl_bias = nn.Parameter(torch.tensor(feats['gmean'], dtype=torch.float32))
        # 绝对丰度损失权重（自适应）
        self.log_var_y = nn.Parameter(torch.tensor(0.0))
        self.log_var_ctrl = nn.Parameter(torch.tensor(0.0))
        self.log_var_delta = nn.Parameter(torch.tensor(0.0))
        self.log_var_fc = nn.Parameter(torch.tensor(0.0))

        self.register_buffer('strain_avg', torch.zeros(16))

    def set_strain_avg(self):
        self.strain_avg.copy_(self.strain_emb.weight.mean(dim=0))

    def _get_strain_emb(self, strain_id):
        base = self.strain_emb(strain_id.clamp(min=0))
        unseen = (strain_id < 0).float().unsqueeze(1)
        return (1 - unseen) * base + unseen * self.strain_avg.unsqueeze(0)

    def ctrl_predict(self, x):
        """状态模型：预测匹配对照（不含化合物信息）"""
        strain_id, chem_id, chem_hash, med, temp, tfeat, sm_id, ct_id = x['bio']
        src_id, ins_id, plt_id = x['ctx']
        cp = self.ctx_prior_proj(x['ctx_prior'])
        se = self._get_strain_emb(strain_id)
        sg = self.strain_genome_proj(x['strain_dist_vec'])
        sme = self.sm_emb(sm_id.clamp(min=0))
        if temp.dim() == 1:
            temp = temp.unsqueeze(1)
        cxt = torch.cat([self.src_emb(src_id.clamp(min=0)), self.ins_emb(ins_id.clamp(min=0)),
                         self.plt_emb(plt_id.clamp(min=0))], dim=1)
        h = self.ctrl_enc(torch.cat([se, sg, med, temp, tfeat, sme, cp, cxt], dim=1))
        return self.fc_ctrl(h) + self.ctrl_bias

    def delta_predict(self, x):
        """扰动模型：低秩 Δ̂ = U @ z_delta + 残差 + GO/ESM2 先验"""
        strain_id, chem_id, chem_hash, med, temp, tfeat, sm_id, ct_id = x['bio']
        chem_seen, strain_seen = x['seen']
        cp = self.ctx_prior_proj(x['ctx_prior'])
        se = self._get_strain_emb(strain_id)
        sg = self.strain_genome_proj(x['strain_dist_vec'])
        ce = self.chem_emb(chem_id.clamp(min=0)) * (chem_id >= 0).float().unsqueeze(1)
        morgan = self.morgan_proj(x['chem_morgan'])
        sme = self.sm_emb(sm_id.clamp(min=0))
        cte = self.ct_emb(ct_id.clamp(min=0))
        if temp.dim() == 1:
            temp = temp.unsqueeze(1)

        z = self.resp_enc(torch.cat([se, sg, ce, morgan, self.desc_proj(x['chem_desc']),
                                     med, temp, tfeat, cte, cp], dim=1))
        # 低秩主项
        delta_lr = torch.mm(z, self.U.T)  # (N, P)
        if self.gate_mode == 'rel':
            # ★ 步骤12 可靠性门控（gpt2 P1-4）：由相似度/支持数驱动，替代硬 seen 门控
            chem_sim = x['chem_max_sim'].unsqueeze(1)
            chem_sp = (x['chem_support'] / 6.3).unsqueeze(1)
            strain_sp = (x['strain_support'] / 7.3).unsqueeze(1)
            g_c = torch.sigmoid(self.gate_c) * (0.2 + 0.8 * chem_seen.unsqueeze(1)
                                                + 0.5 * chem_sim * (1 - chem_seen.unsqueeze(1)))
            g_s = torch.sigmoid(self.gate_s) * (0.3 + 0.7 * strain_seen.unsqueeze(1)
                                                + 0.3 * strain_sp * strain_seen.unsqueeze(1))
        else:
            # 硬门控：unseen 化合物保留 20%，unseen 菌株保留 30%（共享应激响应不归零）
            g_c = torch.sigmoid(self.gate_c) * (0.2 + 0.8 * chem_seen).unsqueeze(1)
            g_s = torch.sigmoid(self.gate_s) * (0.3 + 0.7 * strain_seen).unsqueeze(1)
        delta_lr = g_c * g_s * delta_lr
        # 残差头（低权重，只对已见实体生效）
        resid = self.resid_scale * self.resid_head(z) * (g_c * g_s)
        # GO + ESM2 先验（功能层面响应方向）
        wz = self.w_esm(z)
        effect = self.go_effect(z)
        prior = (torch.sigmoid(self.gate_esm) * torch.mm(wz, self.esm2_emb.T)
                 + torch.sigmoid(self.gate_go) * torch.mm(effect, self.go_mat))
        prior = g_c * g_s * prior
        return delta_lr + resid + prior

    def forward(self, x):
        y_ctrl = self.ctrl_predict(x)
        delta = self.delta_predict(x)
        return y_ctrl + delta

    def components(self, x):
        return self.ctrl_predict(x), self.delta_predict(x)
