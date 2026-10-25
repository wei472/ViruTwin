# -*- coding: utf-8 -*-
"""
Model definitions for ViruTwin classifiers, reconstructed from the training
scripts (09_dl_baselines.py, 10_dpf_fusion.py) and manuscript Table 4.
app.py loads this file via VT_MODELS_PY and calls build_model().
"""
import torch
import torch.nn as nn


class IterLowRank(nn.Module):
    """低秩投影 + GELU + 多步增量迭代 + LayerNorm + 残差"""
    def __init__(self, dh, r=4, T=3):
        super().__init__()
        self.T = T
        self.W1 = nn.ModuleList([nn.Linear(dh, r) for _ in range(T)])
        self.W2 = nn.ModuleList([nn.Linear(r, dh) for _ in range(T)])
        self.norm = nn.LayerNorm(dh)

    def forward(self, h):
        s = h
        for t in range(self.T):
            s = s + self.W2[t](nn.functional.gelu(self.W1[t](s)))
        return self.norm(s) + h


class VFIter(nn.Module):
    """VF-Iter: 特征投影 -> N个IterLowRank块 -> 分类头"""
    def __init__(self, d, h=256, r=8, T=3, n_blocks=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1))
        self.blocks = nn.ModuleList([IterLowRank(h, r, T) for _ in range(n_blocks)])
        self.cls = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 64), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        return self.cls(h)


class DPF(nn.Module):
    """双路融合: 门控 -> 源特异投影 -> 自注意力 -> IterLowRank -> 交叉注意力
    -> 融合 -> alpha平衡 -> 分类头。ablation: None/'gate'/'attn'/'cross'/'alpha'"""
    def __init__(self, d1, d2, h=512, heads=2, r=4, T=2, nb=2, lam=0.1, ablation=None):
        super().__init__()
        assert h == 512
        self.ab = ablation
        if ablation != 'gate':
            self.g1 = nn.Parameter(torch.ones(d1))
            self.g2 = nn.Parameter(torch.ones(d2))
        self.p1 = nn.Sequential(nn.Linear(d1, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1))
        self.p2 = nn.Sequential(nn.Linear(d2, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1))
        if ablation != 'attn':
            self.a1 = nn.MultiheadAttention(32, heads, batch_first=True)
            self.a2 = nn.MultiheadAttention(32, heads, batch_first=True)
        self.b1 = nn.ModuleList([IterLowRank(h, r, T) for _ in range(nb)])
        self.b2 = nn.ModuleList([IterLowRank(h, r, T) for _ in range(nb)])
        if ablation != 'cross':
            self.c1 = nn.MultiheadAttention(32, heads, batch_first=True)
            self.c2 = nn.MultiheadAttention(32, heads, batch_first=True)
            self.lam = lam
        self.fuse = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(0.1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.cls = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 128), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(128, 2))

    def forward(self, x1, x2):
        if self.ab != 'gate':
            x1, x2 = x1 * self.g1, x2 * self.g2
        f1 = self.p1(x1).view(-1, 16, 32)
        f2 = self.p2(x2).view(-1, 16, 32)
        if self.ab != 'attn':
            f1, _ = self.a1(f1, f1, f1)
            f2, _ = self.a2(f2, f2, f2)
        f1, f2 = f1.flatten(1), f2.flatten(1)
        for b in self.b1:
            f1 = b(f1)
        for b in self.b2:
            f2 = b(f2)
        if self.ab != 'cross':
            v1, v2 = f1.view(-1, 16, 32), f2.view(-1, 16, 32)
            o1, _ = self.c1(v1, v2, v2)
            o2, _ = self.c2(v2, v1, v1)
            f1 = f1 + self.lam * o1.flatten(1)
            f2 = f2 + self.lam * o2.flatten(1)
        z = self.fuse(torch.cat([f1, f2], 1))
        a, b = z.chunk(2, 1)
        w = torch.sigmoid(self.alpha)
        final = w * a + (1 - w) * b if self.ab != 'alpha' else a + b
        return self.cls(torch.cat([final, final], 1))


# VF-Iter hyperparameters selected by nested CV (manuscript Table 4)
VFITER_CFG = {
    "vfiter_cat":  dict(d=2304, h=512, r=4,  T=2, n_blocks=2),
    "vfiter_esm2": dict(d=1280, h=256, r=16, T=2, n_blocks=2),
    "vfiter_p5":   dict(d=1024, h=256, r=8,  T=2, n_blocks=2),
}
# DPF ablation mapping (10_dpf_fusion.py); lam=0.1 from matrices/dpf_best.json
DPF_CFG = {
    "dpf":    dict(ablation=None),
    "dpf_m1": dict(ablation='gate'),
    "dpf_m2": dict(ablation='attn'),
    "dpf_m3": dict(ablation='cross'),
    "dpf_m4": dict(ablation='alpha'),
}


def build_model(name, in_dim, state_dict):
    if name in VFITER_CFG:
        m = VFIter(**VFITER_CFG[name])
    elif name in DPF_CFG:
        m = DPF(1280, 1024, lam=0.1, **DPF_CFG[name])
    else:
        raise ValueError(f"no builder for model '{name}'")
    m.load_state_dict(state_dict)
    return m
