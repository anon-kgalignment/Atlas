# sage/modules/models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────
# INFONCE ALIGNMENT LOSS
# ─────────────────────────────────────────────

def infonce_loss(
    A_src       : torch.Tensor,
    A_tgt       : torch.Tensor,
    temperature : float = 0.07,
) -> Tuple[torch.Tensor, float, float]:
    """
    InfoNCE contrastive alignment loss.

    For each seed pair (i, j):
        - A_src[i] and A_tgt[j] are the positive pair
        - All other entities in the batch are negatives

    With batch size N:
        - Current margin loss uses 5 negatives per entity
        - InfoNCE uses N-1 negatives per entity
        - With N=512 that is 100x more learning signal

    The temperature controls sharpness:
        - Lower temperature → harder contrast → stronger signal
        - 0.07 is standard for alignment tasks

    Args:
        A_src       : aligned KG1 embeddings [batch, dim]
                      already L2 normalized
        A_tgt       : aligned KG2 embeddings [batch, dim]
                      already L2 normalized
        temperature : softmax temperature (default 0.07)

    Returns:
        loss        : scalar InfoNCE loss
        pos_sim     : average positive pair similarity
        neg_sim     : average negative pair similarity
    """
    assert A_src.shape == A_tgt.shape, \
        f"Shape mismatch: {A_src.shape} vs {A_tgt.shape}"
    A_src = F.normalize(A_src, dim=1)
    A_tgt = F.normalize(A_tgt, dim=1)

    N = A_src.size(0)

    sim = torch.matmul(A_src, A_tgt.t()) / temperature
    
    labels = torch.arange(N, device=A_src.device)

    loss_s2t = F.cross_entropy(sim, labels)

    loss_t2s = F.cross_entropy(sim.t(), labels)

    loss = (loss_s2t + loss_t2s) / 2

    pos_sim = sim.diag().mean().item() * temperature
    neg_sim = (
        (sim.sum() - sim.diag().sum())
        / (N * (N - 1))
    ).item() * temperature

    return loss, pos_sim, neg_sim


# ─────────────────────────────────────────────
# GLOBAL INFONCE LOSS (all KG2 as negatives)
# ─────────────────────────────────────────────

def infonce_global_loss(
    A_src       : torch.Tensor,
    A_tgt_all   : torch.Tensor,
    tgt_ids     : torch.Tensor,
    temperature : float = 0.07,
) -> Tuple[torch.Tensor, float, float]:
    """
    InfoNCE where the denominator runs over ALL KG2 entities.

    Standard InfoNCE with batch=512 gives 511 negatives.
    This version uses every entity in KG2 (~15K) as a negative,
    so hard cases (similar non-matching entities) are always seen.

    Args:
        A_src       : KG1 embeddings for training pairs [B, dim]
        A_tgt_all   : ALL KG2 embeddings               [N2, dim]
        tgt_ids     : index of positive in A_tgt_all   [B] (long)
        temperature : softmax temperature

    Returns:
        loss, pos_sim, neg_sim
    """
    A_src     = F.normalize(A_src,     dim=1)
    A_tgt_all = F.normalize(A_tgt_all, dim=1)

    # [B, N2] — each src vs every KG2 entity
    sim_global = torch.matmul(A_src, A_tgt_all.t()) / temperature
    loss_s2t   = F.cross_entropy(sim_global, tgt_ids)

    # Symmetric: each positive in KG2 vs every training KG1 embedding
    A_tgt_batch = A_tgt_all[tgt_ids]                         # [B, dim]
    sim_batch   = torch.matmul(A_tgt_batch, A_src.t()) / temperature
    labels      = torch.arange(A_src.size(0), device=A_src.device)
    loss_t2s    = F.cross_entropy(sim_batch, labels)

    loss    = (loss_s2t + loss_t2s) / 2
    pos_sim = (A_src * A_tgt_batch).sum(dim=1).mean().item()
    neg_sim = (sim_global.sum() - sim_global[labels, tgt_ids].sum()).item() / (
        A_src.size(0) * (A_tgt_all.size(0) - 1)
    ) * temperature

    return loss, pos_sim, neg_sim


# ─────────────────────────────────────────────
# TRIPLE COHERENCE LOSS
# ─────────────────────────────────────────────

def triple_coherence_loss(
    aligned_emb     : torch.Tensor,
    relation_emb    : torch.Tensor,
    triples_idx     : torch.Tensor,
    model_type      : str = "TransE",
) -> torch.Tensor:
    """
    Triple coherence loss.

    This is the novel component of SAGE that directly
    solves the link prediction preservation problem.

    The problem:
        Before alignment: vec(Paris) + vec(capital_of) = vec(France)
        After alignment:  vec(Paris) moves, vec(France) moves
                          vec(capital_of) stays fixed
        Result:           vec(Paris_new) + vec(capital_of) ≠ vec(France_new)
        Link prediction breaks.

    The solution:
        During alignment training, penalize violations of
        the KGE model scoring equation.

        For TransE:
            penalty = ||h_aligned + r - t_aligned||
            This forces Paris and France to move together
            so the equation remains approximately satisfied.

        For DistMult:
            penalty = relu(1 - score(h, r, t))
            Forces aligned triples to have high scores.

        For ComplEx:
            penalty based on Hermitian dot product.

    Args:
        aligned_emb  : current aligned embeddings [n_entities, dim]
                       these are the embeddings being optimized
        relation_emb : relation embeddings [n_relations, dim]
                       these are FROZEN from original DICE model
        triples_idx  : sampled triples as indices [batch, 3]
                       columns: [head_idx, relation_idx, tail_idx]
        model_type   : KGE model type
                       "TransE", "DistMult", "ComplEx", "DualE"
                       "Pykeen_TransE" mapped to "TransE"

    Returns:
        scalar coherence loss
    """
    mt = model_type.lower()
    if "transe" in mt:
        mt = "transe"
    elif "distmult" in mt:
        mt = "distmult"
    elif "complex" in mt:
        mt = "complex"
    elif "duale" in mt:
        mt = "duale"
    else:
        mt = "transe" 

    h_idx = triples_idx[:, 0]
    r_idx = triples_idx[:, 1]
    t_idx = triples_idx[:, 2]

    h = aligned_emb[h_idx]   
    r = relation_emb[r_idx]   
    t = aligned_emb[t_idx]    

    if mt == "transe":
        return _transe_coherence(h, r, t)
    elif mt == "distmult":
        return _distmult_coherence(h, r, t)
    elif mt == "complex":
        return _complex_coherence(h, r, t)
    elif mt == "duale":
        return _transe_coherence(h, r, t)
    else:
        return _transe_coherence(h, r, t)


def _transe_coherence(
    h : torch.Tensor,
    r : torch.Tensor,
    t : torch.Tensor,
) -> torch.Tensor:
    """
    TransE coherence loss.

    TransE scoring: h + r ≈ t
    Violation: ||h + r - t|| should be small

    After alignment we want:
        h_aligned + r ≈ t_aligned
    So we penalize large values of ||h + r - t||

    Args:
        h, r, t : head, relation, tail embeddings [batch, dim]

    Returns:
        scalar loss
    """
    residual = h + r - t  

    loss = torch.norm(residual, dim=1).mean()

    return loss


def _distmult_coherence(
    h : torch.Tensor,
    r : torch.Tensor,
    t : torch.Tensor,
) -> torch.Tensor:
    """
    DistMult coherence loss.

    DistMult scoring: sum(h * r * t) should be high

    After alignment we want:
        sum(h_aligned * r * t_aligned) to remain high

    Args:
        h, r, t : head, relation, tail embeddings [batch, dim]

    Returns:
        scalar loss
    """
    score = (h * r * t).sum(dim=1) 

    loss = F.relu(1.0 - score).mean()

    return loss


def _complex_coherence(
    h : torch.Tensor,
    r : torch.Tensor,
    t : torch.Tensor,
) -> torch.Tensor:
    """
    ComplEx coherence loss.

    ComplEx uses complex-valued embeddings.
    The embedding vector is split into real and imaginary parts.

    ComplEx scoring (Hermitian dot product):
        score = Re(h * r * conj(t))
              = sum(h_re*r_re*t_re
                  + h_re*r_im*t_im
                  + h_im*r_re*t_im
                  - h_im*r_im*t_re)

    After alignment we want this score to remain high.

    Args:
        h, r, t : embeddings [batch, dim]
                  first half = real part
                  second half = imaginary part

    Returns:
        scalar loss
    """
    d = h.shape[1] // 2

    h_re, h_im = h[:, :d], h[:, d:]
    r_re, r_im = r[:, :d], r[:, d:]
    t_re, t_im = t[:, :d], t[:, d:]

    score = (
        h_re * r_re * t_re
        + h_re * r_im * t_im
        + h_im * r_re * t_im
        - h_im * r_im * t_re
    ).sum(dim=1) 
    loss = F.relu(1.0 - score).mean()

    return loss


# ─────────────────────────────────────────────
# TRIPLE SAMPLER
# ─────────────────────────────────────────────

def sample_triples_batch(
    triples     : List[Tuple[str, str, str]],
    entity2id   : Dict[str, int],
    relation2id : Dict[str, int],
    batch_size  : int = 256,
    device      : torch.device = torch.device("cpu"),
) -> Optional[torch.Tensor]:
    """
    Sample a random batch of triples and convert
    URI strings to integer indices.

    Args:
        triples     : list of (h_uri, r_uri, t_uri)
        entity2id   : {uri: int} entity index mapping
        relation2id : {uri: int} relation index mapping
        batch_size  : number of triples to sample
        device      : torch device

    Returns:
        tensor of shape [batch, 3] with integer indices
        or None if no valid triples found
    """
    if len(triples) == 0:
        return None
    
    n = len(triples)
    sample_idx = np.random.choice(
        n,
        size=min(batch_size * 3, n),
        replace=False
    )

    batch = []
    for i in sample_idx:
        h_uri, r_uri, t_uri = triples[i]

        h_id = entity2id.get(h_uri)
        r_id = relation2id.get(r_uri)
        t_id = entity2id.get(t_uri)

        if h_id is None or r_id is None or t_id is None:
            continue

        batch.append([h_id, r_id, t_id])

        if len(batch) >= batch_size:
            break

    if len(batch) == 0:
        return None

    return torch.tensor(
        batch,
        dtype=torch.long,
        device=device
    )
    
# ─────────────────────────────────────────────
# MI DISCRIMINATOR + JSD LOSS
# ─────────────────────────────────────────────

class MIDiscriminator(nn.Module):
    """
    Small MLP discriminator for mutual information maximization.

    Takes a pair of entity representations (one from each KG) and
    predicts whether they are a true aligned pair or a fake pair.
    Trained jointly with the encoder via a JSD-MI loss.

    Applied on Z (fused representations before projection) to give
    a cross-KG alignment signal at an intermediate layer, complementary
    to the InfoNCE loss at the final projected layer A.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h1, h2], dim=1)).squeeze(-1)


def mi_jsd_loss(
    discriminator : "MIDiscriminator",
    Z1_pairs      : torch.Tensor,
    Z2_pairs      : torch.Tensor,
) -> torch.Tensor:
    """
    Jensen-Shannon MI lower bound (Deep InfoMax style).

    Positive pairs: (Z1[e], Z2[e]) for training-aligned entities.
    Negative pairs: (Z1[e], Z2[shuffled]) — random cross-KG pairs.

    The encoder is trained to maximize MI between matched entities
    in the two KGs. This forces the intermediate representations
    to be cross-KG aligned, not just at the output layer.

    Args:
        discriminator : MIDiscriminator instance
        Z1_pairs      : fused KG1 representations [B, hidden_dim]
        Z2_pairs      : fused KG2 representations [B, hidden_dim]

    Returns:
        scalar JSD-MI loss (minimized, lower = more MI)
    """
    pos_scores = discriminator(Z1_pairs, Z2_pairs)

    perm = torch.randperm(Z2_pairs.size(0), device=Z2_pairs.device)
    Z2_neg = Z2_pairs[perm]
    neg_scores = discriminator(Z1_pairs, Z2_neg)

    loss = F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()
    return loss


def geometry_consistency_loss(
    A           : torch.Tensor,
    rel_emb     : torch.Tensor,
    triples_idx : torch.Tensor,
    model_type  : str = "TransE",
) -> torch.Tensor:
    """
    Stronger version of triple coherence.
    Penalizes both the absolute violation AND
    the relative change from original geometry.

    For TransE: we want ||h + r - t|| to be
    as small as it was in the original DICE model.
    """
    h_idx = triples_idx[:, 0]
    r_idx = triples_idx[:, 1]
    t_idx = triples_idx[:, 2]

    h = A[h_idx]
    r = rel_emb[r_idx]
    t = A[t_idx]

    h_n = F.normalize(h, dim=1)
    t_n = F.normalize(t, dim=1)

    residual = h_n + r - t_n

    l2_loss = torch.norm(residual, dim=1).mean()
    l1_loss = torch.abs(residual).sum(dim=1).mean()

    return 0.5 * l2_loss + 0.5 * l1_loss