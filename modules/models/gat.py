
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional



def _sparse_softmax(
    scores      : torch.Tensor,
    dst_indices : torch.Tensor,
    n_nodes     : int,
) -> torch.Tensor:
    n_heads = scores.size(1)

    max_scores = torch.full(
        (n_nodes, n_heads),
        float('-inf'),
        device=scores.device,
        dtype=scores.dtype
    )

    dst_exp = dst_indices.unsqueeze(1).expand(-1, n_heads)

    max_scores = max_scores.scatter_reduce(
        0, dst_exp, scores,
        reduce="amax",
        include_self=True
    )

    scores_shifted = scores - max_scores[dst_indices]
    exp_scores = torch.exp(scores_shifted)

    sum_exp = torch.zeros(
        n_nodes, n_heads,
        device=scores.device,
        dtype=scores.dtype
    )
    sum_exp = sum_exp.scatter_add(0, dst_exp, exp_scores)
    sum_exp = sum_exp.clamp(min=1e-8)

    weights = exp_scores / sum_exp[dst_indices]
    return weights


# ─────────────────────────────────────────────
# RELATIONAL GAT LAYER
# ─────────────────────────────────────────────

class RelationalGATLayer(nn.Module):
    """
    One layer of a Relational Graph Attention Network.

    Accepts an optional ext_rel_emb to allow weight sharing
    (siamese mode): the core weights are shared across KGs
    while each KG supplies its own relation embeddings.
    """

    REL_EMBED_DIM = 32

    def __init__(
        self,
        in_dim       : int,
        out_dim      : int,
        n_relations  : int,
        n_heads      : int = 4,
        dropout      : float = 0.1,
        align_rel_id : Optional[int] = None,
    ):
        super().__init__()

        assert out_dim % n_heads == 0
        self.out_dim      = out_dim
        self.n_heads      = n_heads
        self.head_dim     = out_dim // n_heads
        self.align_rel_id = align_rel_id

        # Internal rel embeddings used only when ext_rel_emb is None
        self.rel_embeddings = nn.Embedding(n_relations, self.REL_EMBED_DIM)

        self.W_msg  = nn.Linear(in_dim + self.REL_EMBED_DIM, out_dim, bias=False)
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)

        self.attn_vec = nn.Parameter(torch.Tensor(1, n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.view(1, n_heads, self.head_dim))

        self.align_gate = nn.Parameter(torch.tensor(0.5))
        self.layer_norm = nn.LayerNorm(out_dim)
        self.drop       = nn.Dropout(dropout)

    def forward(
        self,
        x           : torch.Tensor,
        adj_lists   : Dict[int, np.ndarray],
        device      : torch.device,
        ext_rel_emb : Optional[nn.Embedding] = None,
    ) -> torch.Tensor:
        """
        Args:
            ext_rel_emb : if given, use instead of self.rel_embeddings
                          (enables siamese/shared-weight mode)
        """
        rel_source = ext_rel_emb if ext_rel_emb is not None else self.rel_embeddings
        n_entities = x.size(0)
        out = self.W_self(x)

        all_messages = []

        for r_id, edges in adj_lists.items():
            if len(edges) == 0:
                continue

            edges_t = torch.tensor(edges, dtype=torch.long, device=x.device)
            dst = edges_t[:, 0]
            src = edges_t[:, 1]
            n_edges = dst.size(0)

            r_id_t         = torch.tensor(r_id, device=x.device)
            r_emb          = rel_source(r_id_t)
            r_emb_expanded = r_emb.unsqueeze(0).expand(n_edges, -1)

            msg_input  = torch.cat([x[src], r_emb_expanded], dim=1)
            msg        = self.W_msg(msg_input)
            msg_heads  = msg.view(n_edges, self.n_heads, self.head_dim)

            attn_score  = (msg_heads * self.attn_vec).sum(dim=-1)
            attn_score  = F.leaky_relu(attn_score, negative_slope=0.2)
            attn_weight = _sparse_softmax(attn_score, dst, n_entities)
            attn_weight = self.drop(attn_weight)

            if r_id == self.align_rel_id:
                attn_weight = attn_weight * torch.sigmoid(self.align_gate)

            weighted = (msg_heads * attn_weight.unsqueeze(-1)).view(n_edges, self.out_dim)
            all_messages.append((dst, weighted))

        if all_messages:
            agg = torch.zeros(n_entities, self.out_dim, device=x.device, dtype=x.dtype)
            for dst, weighted in all_messages:
                agg = agg.scatter_add(
                    0, dst.unsqueeze(1).expand(-1, self.out_dim), weighted
                )
            out = out + agg

        out = self.layer_norm(out)
        out = F.elu(out)
        return out


# ─────────────────────────────────────────────
# SEPARATE GAT ENCODER (retained for ablation)
# ─────────────────────────────────────────────

class RelationalGATEncoder(nn.Module):
    """
    Two-layer Relational GAT encoder — each KG has its own independent encoder.
    Kept for ablation comparison against SiameseRelationalGATEncoder.
    """

    def __init__(
        self,
        dice_dim     : int,
        labse_dim    : int,
        hidden_dim   : int,
        n_relations  : int,
        n_heads      : int = 4,
        dropout      : float = 0.1,
        align_rel_id : Optional[int] = None,
    ):
        super().__init__()

        input_dim = dice_dim + labse_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.gat1 = RelationalGATLayer(hidden_dim, hidden_dim, n_relations, n_heads, dropout, align_rel_id)
        self.gat2 = RelationalGATLayer(hidden_dim, hidden_dim, n_relations, n_heads, dropout, align_rel_id)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, E, P, adj_lists, device):
        x  = self.input_proj(torch.cat([E, P], dim=1))
        h1 = self.gat1(x, adj_lists, device)
        h1 = h1 + x
        h2 = self.gat2(h1, adj_lists, device)
        h2 = h2 + h1
        return self.out_norm(h2)


# ─────────────────────────────────────────────
# SIAMESE GAT ENCODER  ← MAIN ARCHITECTURE
# ─────────────────────────────────────────────

class SiameseRelationalGATEncoder(nn.Module):
    """
    Siamese Relational GAT encoder.

    Both KG1 and KG2 are processed by the SAME W_msg / W_self /
    attn_vec / layer_norm weights.  Each KG keeps its own
    relation-type embeddings because DBpedia and Wikidata have
    completely different relation vocabularies.

    Architecture (per KG):
        [DICE(256) || E5(1024)] → shared_input_proj → 256
        → shared_gat_layer_1 (with KG-specific rel_emb_1) + residual
        → shared_gat_layer_2 (with KG-specific rel_emb_2) + residual
        → shared_out_norm → 256
    """

    REL_EMBED_DIM = 32

    def __init__(
        self,
        dice_dim         : int,
        labse_dim        : int,
        hidden_dim       : int,
        n_relations_kg1  : int,
        n_relations_kg2  : int,
        n_heads          : int = 4,
        dropout          : float = 0.1,
    ):
        super().__init__()

        input_dim = dice_dim + labse_dim

        # ── Shared input projection ───────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # ── Shared GAT layers ────────────────────────
        self.shared_gat1 = RelationalGATLayer(
            hidden_dim, hidden_dim, n_relations=1,
            n_heads=n_heads, dropout=dropout,
        )
        self.shared_gat2 = RelationalGATLayer(
            hidden_dim, hidden_dim, n_relations=1,
            n_heads=n_heads, dropout=dropout,
        )

        # ── KG-specific relation embeddings ──────────
        self.rel_emb_kg1_1 = nn.Embedding(n_relations_kg1, self.REL_EMBED_DIM)
        self.rel_emb_kg2_1 = nn.Embedding(n_relations_kg2, self.REL_EMBED_DIM)
        self.rel_emb_kg1_2 = nn.Embedding(n_relations_kg1, self.REL_EMBED_DIM)
        self.rel_emb_kg2_2 = nn.Embedding(n_relations_kg2, self.REL_EMBED_DIM)

        self.out_norm = nn.LayerNorm(hidden_dim)

    def _encode(self, E, P, adj_lists, device, rel_emb_l1, rel_emb_l2):
        x  = self.input_proj(torch.cat([E, P], dim=1))
        h1 = self.shared_gat1(x,  adj_lists, device, ext_rel_emb=rel_emb_l1)
        h1 = h1 + x
        h2 = self.shared_gat2(h1, adj_lists, device, ext_rel_emb=rel_emb_l2)
        h2 = h2 + h1
        return self.out_norm(h2)

    def forward_kg1(self, E, P, adj_lists, device):
        return self._encode(E, P, adj_lists, device,
                            self.rel_emb_kg1_1, self.rel_emb_kg1_2)

    def forward_kg2(self, E, P, adj_lists, device):
        return self._encode(E, P, adj_lists, device,
                            self.rel_emb_kg2_1, self.rel_emb_kg2_2)
