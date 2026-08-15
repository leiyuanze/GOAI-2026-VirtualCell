# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 04 模型定义（v4.1：GO通路注意力(unseen门控)+蛋白先验）
基于 v2.9（交互项 + 自适应loss），新增：
  1) Morgan 指纹（2048维 -> 64维投影）替换/增强化合物侧，unseen 化合物获得真实化学结构
  2) ESM2 蛋白 embedding（4422x64）作为输出层蛋白序列先验通路 out_esm = wz @ esm2_emb.T
外部数据来源：PubChem(RDKit Morgan fingerprint), ESM2(Meta)
"""
import numpy as np
import torch
import torch.nn as nn

class VCellModel(nn.Module):
    def __init__(self, feats, P=4422, hidden=256, latent=64, esm_dim=64):
        super().__init__()
        self.P = P
        self.latent = latent

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

        # ctx_prior 投影
        self.ctx_prior_proj = nn.Sequential(
            nn.Linear(P, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64),
        )

        # ★ Morgan 指纹（已 PCA 到 64 维，轻量投影）
        self.morgan_proj = nn.Sequential(
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
        )

        # 混合编码（B/S）：[16+32+64+2+1+3+4+8+64] = 194（Morgan 替换 chem_hash）
        d_mix = 16 + 32 + 64 + 2 + 1 + 3 + 4 + 8 + 64
        self.enc = nn.Sequential(
            nn.Linear(d_mix, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.GELU(),
        )
        # C 分支：化合物侧 [32+64+2+1+3+8] = 110
        d_chem = 32 + 64 + 2 + 1 + 3 + 8
        self.enc_C = nn.Sequential(
            nn.Linear(d_chem, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.GELU(),
        )
        # T 分支：菌株侧 90
        d_strain = 16 + 64 + 2 + 1 + 3 + 4
        self.enc_T = nn.Sequential(
            nn.Linear(d_strain, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.GELU(),
        )

        self.fc_B = nn.Linear(latent, latent)
        self.fc_S = nn.Linear(latent, latent)
        self.fc_C = nn.Linear(latent, latent)
        self.fc_T = nn.Linear(latent, latent)

        # 交互项
        self.interact_mlp = nn.Sequential(
            nn.Linear(latent * 3, latent), nn.LayerNorm(latent), nn.GELU(),
            nn.Linear(latent, latent),
        )
        self.gate_i = nn.Parameter(torch.tensor(1.0))
        self.gate_c = nn.Parameter(torch.tensor(3.0))
        self.gate_s = nn.Parameter(torch.tensor(3.0))

        self.proj = nn.Linear(latent, P)
        self.bias = nn.Parameter(torch.tensor(feats['gmean'], dtype=torch.float32))
        self.calib = nn.Sequential(nn.Linear(4 + 4 + 16, 128), nn.GELU(), nn.Linear(128, P))

        # ★ ESM2 蛋白序列先验通路（带门控，初始小值防止干扰 proj）
        self.register_buffer('esm2_emb', torch.tensor(feats['esm2_emb'], dtype=torch.float32))  # 4422 x 64
        self.w_esm = nn.Linear(latent, esm_dim)  # latent(64) -> esm(64)
        self.gate_esm = nn.Parameter(torch.tensor(0.1))

        # ★ GO 通路注意力（教程 5.4.4 通路归属）：同一通路蛋白共享权重偏置
        self.register_buffer('go_mat', torch.tensor(feats['go_mat'], dtype=torch.float32))  # K x P
        self.go_effect = nn.Linear(latent, feats['n_go'])  # latent -> K 通路效应
        self.gate_go = nn.Parameter(torch.tensor(0.1))

        # 自适应 loss 权重
        self.log_var_mse = nn.Parameter(torch.tensor(0.0))
        self.log_var_fc = nn.Parameter(torch.tensor(0.0))
        self.log_var_ctx = nn.Parameter(torch.tensor(0.0))
        self.log_var_drug = nn.Parameter(torch.tensor(0.0))

        self.register_buffer('strain_avg', torch.zeros(16))

    def set_strain_avg(self):
        self.strain_avg.copy_(self.strain_emb.weight.mean(dim=0))

    def _get_strain_emb(self, strain_id):
        base = self.strain_emb(strain_id.clamp(min=0))
        unseen = (strain_id < 0).float().unsqueeze(1)
        return (1 - unseen) * base + unseen * self.strain_avg.unsqueeze(0)

    def forward(self, x, with_components=False):
        strain_id, chem_id, chem_hash, med, temp, tfeat, sm_id, ct_id = x['bio']
        src_id, ins_id, plt_id = x['ctx']
        chem_seen, strain_seen = x['seen']
        ctx_prior = x['ctx_prior']
        chem_morgan = x['chem_morgan']

        se = self._get_strain_emb(strain_id)
        ce = self.chem_emb(chem_id.clamp(min=0)) * (chem_id >= 0).float().unsqueeze(1)
        sme = self.sm_emb(sm_id.clamp(min=0)); cte = self.ct_emb(ct_id.clamp(min=0))
        if temp.dim() == 1: temp = temp.unsqueeze(1)
        cp = self.ctx_prior_proj(ctx_prior)
        morgan = self.morgan_proj(chem_morgan)  # (N, 64)

        bio_mix = torch.cat([se, ce, morgan, med, temp, tfeat, sme, cte, cp], dim=1)
        h = self.enc(bio_mix)
        bio_c = torch.cat([ce, morgan, med, temp, tfeat, cte], dim=1)
        hc = self.enc_C(bio_c)
        bio_t = torch.cat([se, cp, med, temp, tfeat, sme], dim=1)
        ht = self.enc_T(bio_t)

        b, s = self.fc_B(h), self.fc_S(h)
        c, t = self.fc_C(hc), self.fc_T(ht)

        interact = self.interact_mlp(torch.cat([hc * ht, hc, ht], dim=-1))
        g_i = torch.sigmoid(self.gate_i) * chem_seen.unsqueeze(1) * strain_seen.unsqueeze(1)
        g_c = torch.sigmoid(self.gate_c) * (0.2 + 0.8 * chem_seen).unsqueeze(1)
        g_s = torch.sigmoid(self.gate_s) * strain_seen.unsqueeze(1)
        z = b + s + g_c * c + g_s * t + g_i * interact

        # ★ ESM2 通路（带门控）：wz @ esm2_emb.T 给每个蛋白一个序列先验贡献
        wz = self.w_esm(z)  # (N, 64)
        # ★ GO 通路注意力（只对 unseen 菌株启用：seen 菌株已学特异响应，不约束）
        effect = self.go_effect(z)  # (N, K)
        go_gate = torch.sigmoid(self.gate_go) * (1 - strain_seen).unsqueeze(1)
        out = (self.proj(z)
               + torch.sigmoid(self.gate_esm) * torch.mm(wz, self.esm2_emb.T)
               + go_gate * torch.mm(effect, self.go_mat)
               + self.bias)

        cxt = torch.cat([self.src_emb(src_id.clamp(min=0)), self.ins_emb(ins_id.clamp(min=0)),
                         self.plt_emb(plt_id.clamp(min=0))], dim=1)
        out = out + self.calib(cxt)
        if with_components:
            return out, self.proj(g_c * c), self.proj(g_s * t)
        return out

    def components(self, x):
        strain_id, chem_id, chem_hash, med, temp, tfeat, sm_id, ct_id = x['bio']
        chem_seen, strain_seen = x['seen']
        ctx_prior = x['ctx_prior']
        chem_morgan = x['chem_morgan']

        se = self._get_strain_emb(strain_id)
        ce = self.chem_emb(chem_id.clamp(min=0)) * (chem_id >= 0).float().unsqueeze(1)
        sme = self.sm_emb(sm_id.clamp(min=0)); cte = self.ct_emb(ct_id.clamp(min=0))
        if temp.dim() == 1: temp = temp.unsqueeze(1)
        cp = self.ctx_prior_proj(ctx_prior)
        morgan = self.morgan_proj(chem_morgan)

        bio_mix = torch.cat([se, ce, morgan, med, temp, tfeat, sme, cte, cp], dim=1)
        h = self.enc(bio_mix)
        bio_c = torch.cat([ce, morgan, med, temp, tfeat, cte], dim=1)
        hc = self.enc_C(bio_c)
        bio_t = torch.cat([se, cp, med, temp, tfeat, sme], dim=1)
        ht = self.enc_T(bio_t)

        b, s = self.fc_B(h), self.fc_S(h)
        c, t = self.fc_C(hc), self.fc_T(ht)
        g_c = torch.sigmoid(self.gate_c) * (0.2 + 0.8 * chem_seen).unsqueeze(1)
        g_s = torch.sigmoid(self.gate_s) * strain_seen.unsqueeze(1)
        return self.proj(b), self.proj(s), self.proj(g_c * c), self.proj(g_s * t)

    def loss_weights(self):
        w_mse = torch.exp(-self.log_var_mse)
        w_fc = torch.exp(-self.log_var_fc)
        w_ctx = torch.exp(-self.log_var_ctx)
        w_drug = torch.exp(-self.log_var_drug)
        return (0.5 * w_mse + 0.5 * self.log_var_mse,
                0.25 * w_fc + 0.25 * self.log_var_fc,
                0.25 * w_ctx + 0.25 * self.log_var_ctx,
                0.25 * w_drug + 0.25 * self.log_var_drug)
