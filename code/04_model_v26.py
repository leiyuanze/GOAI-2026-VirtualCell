# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 04 模型定义（v2.6：T分支增强 + unseen门控归零）
基于 v2.5 输入级分离，两个关键改进：
  1) T 分支输入加入 ctx_prior 投影（蛋白空间先验→64维），提升已见菌株调制精度
  2) 门控公式改为 g_s = sigmoid(gate_s) * strain_seen（unseen→0，防止T分支添乱）
保留 v2.5 的 C 分支化合物侧编码、B/S 混合编码、批次校准。
"""
import numpy as np
import torch
import torch.nn as nn

class VCellModel(nn.Module):
    def __init__(self, feats, P=4422, hidden=256, latent=64):
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

        # ctx_prior 投影（蛋白空间→64维，给 T 分支用）
        self.ctx_prior_proj = nn.Sequential(
            nn.Linear(P, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 64),
        )

        # 混合编码（B/S 用）：菌株+化合物+上下文 98 维
        d_mix = 16 + 32 + 32 + 2 + 1 + 3 + 4 + 8
        self.enc = nn.Sequential(
            nn.Linear(d_mix, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.GELU(),
        )
        # C 分支编码：化合物侧 [chem_emb32 + hash32 + med2 + temp1 + time3 + ct8] = 78
        d_chem = 32 + 32 + 2 + 1 + 3 + 8
        self.enc_C = nn.Sequential(
            nn.Linear(d_chem, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.GELU(),
        )
        # T 分支编码：菌株侧 [strain_emb16 + ctx_prior_proj64 + med2 + temp1 + time3 + sm4] = 90
        d_strain = 16 + 64 + 2 + 1 + 3 + 4
        self.enc_T = nn.Sequential(
            nn.Linear(d_strain, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.GELU(),
        )

        self.fc_B = nn.Linear(latent, latent)
        self.fc_S = nn.Linear(latent, latent)
        self.fc_C = nn.Linear(latent, latent)
        self.fc_T = nn.Linear(latent, latent)
        self.gate_c = nn.Parameter(torch.tensor(3.0))
        self.gate_s = nn.Parameter(torch.tensor(3.0))

        self.proj = nn.Linear(latent, P)
        self.bias = nn.Parameter(torch.tensor(feats['gmean'], dtype=torch.float32))
        self.calib = nn.Sequential(nn.Linear(4 + 4 + 16, 128), nn.GELU(), nn.Linear(128, P))

    def forward(self, x, with_components=False):
        strain_id, chem_id, chem_hash, med, temp, tfeat, sm_id, ct_id = x['bio']
        src_id, ins_id, plt_id = x['ctx']
        chem_seen, strain_seen = x['seen']
        ctx_prior = x['ctx_prior']  # (N, P)

        se = self.strain_emb(strain_id.clamp(min=0))
        ce = self.chem_emb(chem_id.clamp(min=0)) * (chem_id >= 0).float().unsqueeze(1)
        sme = self.sm_emb(sm_id.clamp(min=0)); cte = self.ct_emb(ct_id.clamp(min=0))
        if temp.dim() == 1:
            temp = temp.unsqueeze(1)

        # 混合编码（B/S）
        bio_mix = torch.cat([se, ce, chem_hash, med, temp, tfeat, sme, cte], dim=1)
        h = self.enc(bio_mix)
        # C 分支（不含菌株）
        bio_c = torch.cat([ce, chem_hash, med, temp, tfeat, cte], dim=1)
        hc = self.enc_C(bio_c)
        # T 分支（菌株侧 + ctx_prior 投影）
        cp = self.ctx_prior_proj(ctx_prior)  # (N, 64)
        bio_t = torch.cat([se, cp, med, temp, tfeat, sme], dim=1)
        ht = self.enc_T(bio_t)

        b, s = self.fc_B(h), self.fc_S(h)
        c, t = self.fc_C(hc), self.fc_T(ht)
        g_c = torch.sigmoid(self.gate_c) * (0.2 + 0.8 * chem_seen).unsqueeze(1)
        # v2.6 关键改动：unseen 菌株 T 分支贡献 = 0
        g_s = torch.sigmoid(self.gate_s) * strain_seen.unsqueeze(1)
        z = b + s + g_c * c + g_s * t
        out = self.proj(z) + self.bias
        cxt = torch.cat([self.src_emb(src_id.clamp(min=0)), self.ins_emb(ins_id.clamp(min=0)),
                         self.plt_emb(plt_id.clamp(min=0))], dim=1)
        out = out + self.calib(cxt)
        if with_components:
            return out, self.proj(g_c * c), self.proj(g_s * t)
        return out

    def components(self, x):
        """返回 yB/yS/yC/yT 蛋白空间分量"""
        strain_id, chem_id, chem_hash, med, temp, tfeat, sm_id, ct_id = x['bio']
        src_id, ins_id, plt_id = x['ctx']
        chem_seen, strain_seen = x['seen']
        ctx_prior = x['ctx_prior']

        se = self.strain_emb(strain_id.clamp(min=0))
        ce = self.chem_emb(chem_id.clamp(min=0)) * (chem_id >= 0).float().unsqueeze(1)
        sme = self.sm_emb(sm_id.clamp(min=0)); cte = self.ct_emb(ct_id.clamp(min=0))
        if temp.dim() == 1:
            temp = temp.unsqueeze(1)

        bio_mix = torch.cat([se, ce, chem_hash, med, temp, tfeat, sme, cte], dim=1)
        h = self.enc(bio_mix)
        bio_c = torch.cat([ce, chem_hash, med, temp, tfeat, cte], dim=1)
        hc = self.enc_C(bio_c)
        cp = self.ctx_prior_proj(ctx_prior)
        bio_t = torch.cat([se, cp, med, temp, tfeat, sme], dim=1)
        ht = self.enc_T(bio_t)

        b, s = self.fc_B(h), self.fc_S(h)
        c, t = self.fc_C(hc), self.fc_T(ht)
        g_c = torch.sigmoid(self.gate_c) * (0.2 + 0.8 * chem_seen).unsqueeze(1)
        g_s = torch.sigmoid(self.gate_s) * strain_seen.unsqueeze(1)
        return self.proj(b), self.proj(s), self.proj(g_c * c), self.proj(g_s * t)
