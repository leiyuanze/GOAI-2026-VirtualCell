# -*- coding: utf-8 -*-
"""
GOAI 虚拟细胞 · 04 模型定义（v2.3 版：gate 0.05 + latent 128 + 去 hash）
供 v2.3/v2.3m 训练使用；v2.1 定稿版见 04_model_v21.py
"""
import numpy as np
import torch
import torch.nn as nn

class VCellModel(nn.Module):
    def __init__(self, feats, P=4422, hidden=256, latent=128, drop_ent=0.0):
        super().__init__()
        self.P = P
        self.latent = latent
        self.drop_ent = drop_ent

        self.strain_emb = nn.Embedding(feats['n_strains'] + 1, 16)
        self.chem_emb = nn.Embedding(feats['n_chems'] + 1, 32)
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
        self.NEUTRAL_S = feats['n_strains']
        self.NEUTRAL_C = feats['n_chems']

        d_in = 16 + 32 + 2 + 1 + 3 + 4 + 8   # 66 维（去 hash）
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
        self.proj = nn.Linear(latent, P)
        self.bias = nn.Parameter(torch.tensor(feats['gmean'], dtype=torch.float32))
        self.calib = nn.Sequential(nn.Linear(4 + 4 + 16, 128), nn.GELU(), nn.Linear(128, P))

    def _emb(self, x):
        strain_id, chem_id, med, temp, tfeat, sm_id, ct_id = x['bio']
        bs = strain_id.shape[0]
        s_id = strain_id.clamp(min=0, max=self.NEUTRAL_S)
        c_id = torch.where(chem_id >= 0, chem_id, torch.full_like(chem_id, self.NEUTRAL_C))
        se = self.strain_emb(s_id)
        ce = self.chem_emb(c_id) * (chem_id >= 0).float().unsqueeze(1)
        if self.training and self.drop_ent > 0:
            se = se * (torch.rand(bs, 1, device=se.device) > self.drop_ent).float()
            ce = ce * (torch.rand(bs, 1, device=ce.device) > self.drop_ent).float()
        sme = self.sm_emb(sm_id.clamp(min=0)); cte = self.ct_emb(ct_id.clamp(min=0))
        if temp.dim() == 1:
            temp = temp.unsqueeze(1)
        return torch.cat([se, ce, med, temp, tfeat, sme, cte], dim=1)

    def _decompose(self, bio, chem_seen, strain_seen):
        h = self.enc(bio)
        b, s, c, t = self.fc_B(h), self.fc_S(h), self.fc_C(h), self.fc_T(h)
        base_c = 0.2 + 0.8 * chem_seen
        base_s = 0.2 + 0.8 * strain_seen
        g_c = torch.sigmoid(self.gate_c) * base_c.unsqueeze(1)
        g_s = torch.sigmoid(self.gate_s) * base_s.unsqueeze(1)
        return b, s, c, t, g_c, g_s

    def forward(self, x, with_components=False, ret_latent=False):
        chem_seen, strain_seen = x['seen']
        bio = self._emb(x)
        b, s, c, t, g_c, g_s = self._decompose(bio, chem_seen, strain_seen)
        z = b + s + g_c * c + g_s * t
        if ret_latent:
            return z, b, s, c, t, g_c, g_s
        out = self.proj(z) + self.bias
        cxt = torch.cat([self.src_emb(x['ctx'][0].clamp(min=0)), self.ins_emb(x['ctx'][1].clamp(min=0)),
                         self.plt_emb(x['ctx'][2].clamp(min=0))], dim=1)
        out = out + self.calib(cxt)
        if with_components:
            return out, self.proj(g_c * c), self.proj(g_s * t)
        return out

    def neutral_latent(self, x, neutral='strain'):
        strain_id, chem_id, med, temp, tfeat, sm_id, ct_id = x['bio']
        chem_seen, strain_seen = x['seen']
        if neutral == 'strain':
            x2 = {'bio': [torch.full_like(strain_id, self.NEUTRAL_S), chem_id, med, temp, tfeat, sm_id, ct_id],
                  'ctx': x['ctx'], 'seen': [chem_seen, torch.zeros_like(strain_seen)]}
        else:
            x2 = {'bio': [strain_id, torch.full_like(chem_id, -1), med, temp, tfeat, sm_id, ct_id],
                  'ctx': x['ctx'], 'seen': [torch.zeros_like(chem_seen), strain_seen]}
        z, *_ = self.forward(x2, ret_latent=True)
        return z
