USE_DST_ONLY = False
USE_NEURO_SYMBOLIC = True
assert USE_DST_ONLY ^ USE_NEURO_SYMBOLIC, "Enable exactly one fusion variant."

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
device = torch.device("cpu")
from modules.belief.unsure.unsure.boe import BOE


def normalize_feature(x):
    # stable normalization that preserves shape and differences
    return x / (x.max() + 1e-8)



class AdaptiveFusionMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8):
        super().__init__()
        self.sim_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, lb, rl, ty):
        x = torch.stack([lb, rl, ty], dim=1)
        w = torch.softmax(self.sim_mlp(x), dim=1)
        return w

'''
def _make_boe_from_similarity(sim: float):
    sim = float(np.clip(sim, 0.0, 1.0))
    ign = float(np.clip(1.0 - sim, 0.0, 1.0))

    m = BOE(['same', 'not'])
    m.set_mass(('same',), (1.0 - ign) * sim)
    m.set_mass(('not',),  (1.0 - ign) * (1.0 - sim))
    m.set_mass(('same', 'not'), ign)
    return m
'''

def _make_boe_from_similarity(sim: float, alpha=2.0, beta=0.2):
    """
    FuzzyEA-style intuitionistic fuzzy transformation:
    sim -> (μ, ν, π)  then converted into a BOE mass.
    """
    # 1. Keep similarity in valid range
    sim = float(np.clip(sim, 0.0, 1.0))

    # 2. Intuitionistic fuzzy mapping
    mu_raw = sim ** alpha
    nu_raw = (1.0 - sim) ** alpha
    S = mu_raw + nu_raw

    # normalize with β to ensure π > 0 (key for DST stability)
    mu = mu_raw / (S + beta)
    nu = nu_raw / (S + beta)
    pi = 1.0 - mu - nu     # hesitation / ignorance

    pi = max(pi, 0.0)      # numerically guard
    mu = min(max(mu, 0.0), 1.0)
    nu = min(max(nu, 0.0), 1.0)

    # 3. Build BOE mass
    m = BOE(['same', 'not'])
    m.set_mass(('same',), mu)
    m.set_mass(('not',), nu)
    m.set_mass(('same', 'not'), pi)

    return m


def fuse_dst_three(lb, rl, ty, rule="dcr"):
    m_lbl = _make_boe_from_similarity(lb)
    m_rel = _make_boe_from_similarity(rl)
    m_typ = _make_boe_from_similarity(ty)

    # Pairwise conflicts BEFORE fusion
    k12 = m_lbl.conflict(m_rel)
    k13 = m_lbl.conflict(m_typ)
    k23 = m_rel.conflict(m_typ)
    conflict = (k12 + k13 + k23) / 3.0

    # fuse masses (DCR / Yager)
    fused = m_lbl.dcr_multisource([m_rel, m_typ]) if rule == "dcr" \
         else m_lbl.yager_multisource([m_rel, m_typ])

    belief = fused.belief(("same",))
    return belief, conflict



def belief_fusion_loss_hybrid(lb_vec, rl_vec, ty_vec, fusion_module, labels, dst_rule="dcr"):
    device = lb_vec.device

    # ✔ CORRECT: do NOT shrink evidence!
    lb = torch.clamp(lb_vec, 0, 1)
    rl = torch.clamp(rl_vec, 0, 1)
    ty = torch.clamp(ty_vec, 0, 1)

    # 2. Compute DST
    beliefs = []
    conflicts = []
    for i in range(len(lb)):
        b, k = fuse_dst_three(float(lb[i]), float(rl[i]), float(ty[i]), dst_rule)
        beliefs.append(b); conflicts.append(k)

    beliefs   = torch.tensor(beliefs, device=device)
    conflicts = torch.tensor(conflicts, device=device)

    # 3. Trust weights
    w = fusion_module(lb, rl, ty)

    # 4. Labels
    labels = labels.float().to(device)

    # 5. Loss
    mse_belief = ((beliefs - labels)**2).mean()
    entropy = -(w * torch.log(w + 1e-12)).sum(1).mean()
    conflict_penalty = conflicts.mean()

    loss = mse_belief + 0.05*entropy + 0.1*conflict_penalty

    return loss, beliefs, conflicts, w




def belief_fusion_loss_pure(lb_vec, rl_vec, ty_vec, dst_rule='dcr'):
    """
    Pure symbolic DST fusion (no cosine).
    Inputs: lb_vec, rl_vec, ty_vec are 1D tensors of equal length.
    """
    device = lb_vec.device
    beliefs = []
    conflicts = []

    for i in range(len(lb_vec)):
        b, k = fuse_dst_three(
            float(lb_vec[i].item()),
            float(rl_vec[i].item()),
            float(ty_vec[i].item()),
            rule=dst_rule
        )
        beliefs.append(b)
        conflicts.append(k if k is not None else 0.0)

    beliefs   = torch.tensor(beliefs,   dtype=lb_vec.dtype, device=device)
    conflicts = torch.tensor(conflicts, dtype=lb_vec.dtype, device=device)

    # supervision: target belief=1 for matching pairs
    loss = F.mse_loss(beliefs, torch.ones_like(beliefs)) + 0.1 * conflicts.mean()
    return loss, beliefs, conflicts


def _fuse_score(emb, ty, rl, lb):
    # Collect all available sources
    sources = [ty, rl, lb]
    active = [s for s in sources if s != 0]

    # If no similarity signals at all, return 0
    if len(active) == 0:
        return 0.0

    # Simple, fair average (no pre-weighting)
    return float(sum(active) / len(active))





