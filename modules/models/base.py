import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class SharedSpaceAlignmentNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        # vector gating (learned per-dimension)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        z = self.net(x)              
        a = torch.sigmoid(self.alpha)  
        return (1 - a) * x + a * z     # 
    
    
class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.W_rel = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W_rel)
        
    def forward(self, x, adj_lists):
        """
        x: [num_entities, in_dim] base embeddings
        adj_lists: dict {relation_id: list of (i, j) edges}
        """
        out = torch.zeros(x.size(0), self.W_self.out_features, device=x.device)

        # Self-loop
        out += self.W_self(x)

        for r, edges in adj_lists.items():
            if len(edges) == 0: continue
            edges = torch.tensor(edges, dtype=torch.long, device=x.device)
            src = edges[:, 1]  # j → i
            dst = edges[:, 0]  # i
            messages = torch.matmul(x[src], self.W_rel[r])  
            deg = torch.bincount(dst, minlength=x.size(0)).clamp(min=1).float()
            out.index_add_(0, dst, messages / deg[dst].unsqueeze(1))

        return F.relu(out)
    
    
class RGCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, num_relations):
        super().__init__()
        self.layer1 = RGCNLayer(in_dim, hid_dim, num_relations)
        self.layer2 = RGCNLayer(hid_dim, hid_dim, num_relations)

    def forward(self, x, adj_lists):
        h = self.layer1(x, adj_lists)
        h = self.layer2(h, adj_lists)
        return h   # h_struct(i)


    
class Fusion(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2*dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(2*dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)   
        )

    def forward(self, e, h):
        x = torch.cat([e, h], dim=-1)
        z = self.mlp(x)                     
        a = torch.sigmoid(self.gate(x))     
        out = (1 - a) * e + a * z            
        return out


def symmetric_margin_loss(S, T, margin=0.5, k=32):
    S = F.normalize(S, dim=1)
    T = F.normalize(T, dim=1)

    N = S.size(0)
    sim = torch.matmul(S, T.t())  # cosine sim
    mask = torch.eye(N, device=S.device).bool()
    sim_neg = sim.masked_fill(mask, -1e9)

    # hard negatives
    k = min(k, N-1)
    hard_neg, _ = torch.topk(sim_neg, k=k, dim=1)
    neg_mean = hard_neg.mean(dim=1)

    pos = sim.diag()

    # margin loss: pos >= neg + margin
    L_ST = F.relu(neg_mean + margin - pos).mean()

    # symmetric term: T->S
    sim2 = sim.t()
    sim2_neg = sim2.masked_fill(mask, -1e9)
    hard_neg2, _ = torch.topk(sim2_neg, k=k, dim=1)
    neg_mean2 = hard_neg2.mean(dim=1)
    pos2 = sim2.diag()

    L_TS = F.relu(neg_mean2 + margin - pos2).mean()

    return L_ST + L_TS, pos.mean().item(), neg_mean.mean().item()



