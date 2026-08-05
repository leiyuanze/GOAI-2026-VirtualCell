# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 04 模型定义（v2.1 定稿版 —— 与 model_v21.pt 权重匹配）
MLP 主干 + 残差分解（B/S/C/T + 实体可见性门控 0.2+0.8·seen）+ 蛋白模块多头 + 批次校准
架构评审结论已记录：模块多头与单层 Linear 数学等价（v2.2 已验证性能一致）；
gate 0.19 残留、共表达 mask-aware 缺陷均在文档中如实披露为已评估项。
"""
import numpy as np
import torch
import torch.nn as nn

class VCellModel(nn.Module):
    def __init__(self, feats, P=4422, n_modules=64, hidden=256, latent=64):
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

        d_in = 16 + 32 + 32 + 2 + 1 + 3 + 4 + 8   # 98 维（含 hash 32）
        self.enc = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, latent), nn.LayerNorm(latent), nn.GELU(),
        )

        self.fc_B = nn.Linear(latent, latent)
        self.fc_S = nn.Linear(latent, latent)
        self.fc_C = nn.Linear(latent, latent)
        self.fc_T = nn.Linear(latent, latent)
        self.gate_c = nn.Parameter(torch.tensor(3.0))
        self.gate_s = nn.Parameter(torch.tensor(3.0))

        # 蛋白模块多头（与单层 Linear 数学等价，保留原始实现以匹配权重）
        self.module_id = torch.from_numpy(feats['module_id']).long()
        sizes = np.bincount(feats['module_id'], minlength=n_modules)
        self.module_heads = nn.ModuleList([nn.Linear(latent, int(s)) for s in sizes])
        self.bias = nn.Parameter(torch.tensor(feats['gmean'], dtype=torch.float32))

        self.calib = nn.Sequential(nn.Linear(4 + 4 + 16, 128), nn.GELU(), nn.Linear(128, P))

    def forward(self, x, with_components=False):
        strain_id, chem_id, chem_hash, med, temp, tfeat, sm_id, ct_id = x['bio']
        src_id, ins_id, plt_id = x['ctx']
        chem_seen, strain_seen = x['seen']
        se = self.strain_emb(strain_id.clamp(min=0))
        ce = self.chem_emb(chem_id.clamp(min=0)) * (chem_id >= 0).float().unsqueeze(1)
        sme = self.sm_emb(sm_id.clamp(min=0))
        cte = self.ct_emb(ct_id.clamp(min=0))
        if temp.dim() == 1:
            temp = temp.unsqueeze(1)
        bio = torch.cat([se, ce, chem_hash, med, temp, tfeat, sme, cte], dim=1)
        h = self.enc(bio)
        b, s, c, t = self.fc_B(h), self.fc_S(h), self.fc_C(h), self.fc_T(h)
        g_c = torch.sigmoid(self.gate_c) * (0.2 + 0.8 * chem_seen).unsqueeze(1)
        g_s = torch.sigmoid(self.gate_s) * (0.2 + 0.8 * strain_seen).unsqueeze(1)
        z = b + s + g_c * c + g_s * t
        out = self.project(z) + self.bias
        cxt = torch.cat([self.src_emb(src_id.clamp(min=0)), self.ins_emb(ins_id.clamp(min=0)),
                         self.plt_emb(plt_id.clamp(min=0))], dim=1)
        out = out + self.calib(cxt)
        if with_components:
            return out, self.project(g_c * c), self.project(g_s * t)
        return out

    def project(self, z):
        bs = z.shape[0]
        out = torch.zeros(bs, self.P, device=z.device)
        for k, head in enumerate(self.module_heads):
            idx = torch.nonzero(self.module_id == k).squeeze(1)
            out[:, idx] = head(z)
        return out

    def components(self, x):
        """返回 yB/yS/yC/yT 蛋白空间分量（组件级监督用）"""
        strain_id, chem_id, chem_hash, med, temp, tfeat, sm_id, ct_id = x['bio']
        src_id, ins_id, plt_id = x['ctx']
        chem_seen, strain_seen = x['seen']
        se = self.strain_emb(strain_id.clamp(min=0))
        ce = self.chem_emb(chem_id.clamp(min=0)) * (chem_id >= 0).float().unsqueeze(1)
        sme = self.sm_emb(sm_id.clamp(min=0))
        cte = self.ct_emb(ct_id.clamp(min=0))
        if temp.dim() == 1:
            temp = temp.unsqueeze(1)
        bio = torch.cat([se, ce, chem_hash, med, temp, tfeat, sme, cte], dim=1)
        h = self.enc(bio)
        b, s, c, t = self.fc_B(h), self.fc_S(h), self.fc_C(h), self.fc_T(h)
        g_c = torch.sigmoid(self.gate_c) * (0.2 + 0.8 * chem_seen).unsqueeze(1)
        g_s = torch.sigmoid(self.gate_s) * (0.2 + 0.8 * strain_seen).unsqueeze(1)
        return self.project(b), self.project(s), self.project(g_c * c), self.project(g_s * t)
