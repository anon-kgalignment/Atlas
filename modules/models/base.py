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
        z = self.net(x)              # transformed
        a = torch.sigmoid(self.alpha)  # [d] gate
        return (1 - a) * x + a * z     # residual interpolation

        
class CriticAlignmentNN(nn.Module):
    """
    Lightweight critic that refines proposer outputs with a residual mapping.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        # learnable blend like alpha in proposer
        self.beta = nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def forward(self, x):
        z = self.fc2(self.act(self.fc1(x)))
        beta = torch.sigmoid(self.beta)
        x_star = (1 - beta) * x + beta * z
        return x_star, z
    

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

        # Message passing for every relation
        for r, edges in adj_lists.items():
            if len(edges) == 0: continue
            edges = torch.tensor(edges, dtype=torch.long, device=x.device)
            src = edges[:, 1]  # j → i
            dst = edges[:, 0]  # i
            messages = torch.matmul(x[src], self.W_rel[r])
            
            # normalize by number of neighbors
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
            nn.Linear(hidden, dim)   # scalar alpha_i
        )

    def forward(self, e, h):
        x = torch.cat([e, h], dim=-1)

        z = self.mlp(x)                      # transformed mix
        a = torch.sigmoid(self.gate(x))      # (batch, 1)

        out = (1 - a) * e + a * z            # per-entity gating
        return out



#def loss_fn(S_aligned, T_aligned, S_shared, T_shared, S_train_tensor, T_train_tensor, w1, w2, w3, w4):
    #mse_loss = nn.MSELoss()

    #structure_loss = mse_loss(S_aligned, S_train_tensor) + mse_loss(T_aligned, T_train_tensor)
    #alignment_loss = mse_loss(S_shared, T_shared)
    #cosine_sim_loss = mse_loss(nn.functional.cosine_similarity(S_train_tensor, S_aligned, dim=1), torch.ones_like(S_train_tensor[:, 0]))
    #magnitude_loss = mse_loss(torch.norm(S_train_tensor, dim=1), torch.norm(S_aligned, dim=1)) + \
                        #mse_loss(torch.norm(T_train_tensor, dim=1), torch.norm(T_aligned, dim=1))

    # Compute total weight sum (avoid division by zero)
    #weight_sum = w1 + w2 + w3 + w4
    #if weight_sum == 0:
        #return structure_loss + alignment_loss + cosine_sim_loss + magnitude_loss  # Fallback if all weights are 0
        
    # Normalize the loss values by the weight sum
    #total_loss = (w1 * structure_loss + w2 * alignment_loss + w3 * cosine_sim_loss + w4 * magnitude_loss) / weight_sum

    #return total_loss


def loss_fn(
    S_aligned, T_aligned, S_shared, T_shared,
    S_train_tensor, T_train_tensor,
    w1, w2, w3, w4,
    weights=None,            
    return_details=True
):
    mse_loss = nn.MSELoss(reduction='none')  # keep per-pair losses

    # === your cosine loss ===
    def cosine_similarity_loss(original, transformed):
        cos_sim = F.cosine_similarity(original, transformed, dim=1)
        return (1 - cos_sim)  # per-pair dissimilarity

    # === contrastive loss ===
    def contrastive_loss(S_shared, T_shared, margin=1.0, k=20, batch_size=512):
        batch_size_total = S_shared.size(0)
        all_pos, all_mean_neg = [], []
        for start in range(0, batch_size_total, batch_size):
            end = min(start + batch_size, batch_size_total)
            S_batch = S_shared[start:end]
            sim_matrix = F.cosine_similarity(S_batch.unsqueeze(1), T_shared.unsqueeze(0), dim=2)
            pos = sim_matrix[:, start:end].diagonal()
            neg = sim_matrix.clone()
            neg[:, start:end] = -1
            topk_neg, _ = torch.topk(neg, k=min(k, batch_size_total - 1), dim=1)
            mean_neg = topk_neg.mean(dim=1)
            all_pos.append(pos)
            all_mean_neg.append(mean_neg)
        pos_all = torch.cat(all_pos)
        mean_neg_all = torch.cat(all_mean_neg)
        const_loss = F.relu(margin - pos_all + mean_neg_all)
        return const_loss, pos_all, mean_neg_all


    # === compute base losses, per pair ===
    alignment_loss = F.mse_loss(S_shared, T_shared)

    #structure_loss = mse_loss(S_aligned, S_train_tensor).mean(dim=1) + \
                     #mse_loss(T_aligned, T_train_tensor).mean(dim=1)
    contrastive_loss_val, pos_all, mean_neg_all = contrastive_loss(S_shared, T_shared)
    directional_loss = cosine_similarity_loss(S_train_tensor, S_aligned) + \
                       cosine_similarity_loss(T_train_tensor, T_aligned)
    magnitude_loss = torch.abs(torch.norm(S_train_tensor, dim=1) - torch.norm(S_aligned, dim=1)) + \
                     torch.abs(torch.norm(T_train_tensor, dim=1) - torch.norm(T_aligned, dim=1))

    # === combine them per pair ===
    total_loss_per_pair = (
        w1 * alignment_loss +
        w2 * contrastive_loss_val +
        w3 * directional_loss +
        w4 * magnitude_loss
    )

    # === apply belief weighting ===
    if weights is not None:
        weights = weights.to(total_loss_per_pair.device)
        total_loss_per_pair = total_loss_per_pair * weights

    total_loss = total_loss_per_pair.mean()

    if return_details:
        contrastive_mean = contrastive_loss_val.mean().item()
        pos_mean = pos_all.mean().item()
        neg_mean = mean_neg_all.mean().item()
        return total_loss, contrastive_mean, pos_mean, neg_mean
    else:
        return total_loss



def critic_loss_fn(S_hat, T_hat, S_star, T_star):
    """
    Make critic's refined embeddings (S*, T*) agree with proposer (S^, T^) and
    stay mutually close (stability), using MSE + cosine agreement.
    """
    mse = nn.MSELoss()
    # keep critic close to proposer outputs
    agree = mse(S_star, S_hat) + mse(T_star, T_hat)
    # also encourage S* and T* to be close to each other
    cos_ST = F.cosine_similarity(S_star, T_star, dim=1).mean()
    stab = 1.0 - cos_ST
    return agree + stab


def alignment_loss(S, T):
    S = F.normalize(S, p=2, dim=1)
    T = F.normalize(T, p=2, dim=1)
    return (1 - (S * T).sum(dim=1)).mean()



def alignment_contrastive_loss(S, T, margin=1.0, k=128, debug=False):
    S = F.normalize(S, p=2, dim=1)
    T = F.normalize(T, p=2, dim=1)

    N = S.shape[0]      # <<< FIX HERE

    # full cosine similarity matrix
    sim = F.cosine_similarity(S.unsqueeze(1), T.unsqueeze(0), dim=2)

    # mask diagonal
    neg = sim - torch.eye(N, device=S.device) * 2.0

    hard_neg_vals, _ = torch.topk(neg, k=min(k, N-1), dim=1)
    neg_mean = hard_neg_vals.mean(dim=1)

    pos = sim.diag()

    loss_per_sample = F.relu(margin - pos + neg_mean)
    loss = loss_per_sample.mean()

    if debug:
        print("mean diag cosine (true pairs):    ", pos.mean().item())
        print("mean off-diag cosine (negatives): ", neg_mean.mean().item())

    return loss, pos.mean().item(), neg_mean.mean().item()


def alignment_margin_loss(S, T, margin=0.5, k=32, debug=False):
    """
    S : [N, d] source embeddings  
    T : [N, d] target embeddings  
    margin : positive-negative margin  
    k : number of hardest negatives to use  
    """
    # Normalize for cosine similarity
    S = F.normalize(S, dim=1)
    T = F.normalize(T, dim=1)

    N = S.size(0)

    # ---- (1) Full cosine similarity matrix ----
    sim = torch.matmul(S, T.t())   # same as cosine due to normalization

    # ---- (2) Mask diagonal (positive pairs) ----
    mask = torch.eye(N, device=S.device).bool()
    sim_neg = sim.masked_fill(mask, -1e9)

    # ---- (3) Hard negative mining ----
    k = min(k, N - 1)  # safety for small batches
    hard_neg_vals, _ = torch.topk(sim_neg, k=k, dim=1)  # [N, k]
    neg_mean = hard_neg_vals.mean(dim=1)                # [N]

    # ---- (4) Positive scores ----
    pos = sim.diag()                                   # [N]

    # ---- (5) Margin ranking loss ----
    # want: pos >= neg + margin
    loss_per_sample = F.relu(neg_mean + margin - pos)

    loss = loss_per_sample.mean()

    if debug:
        print("Avg positive:", pos.mean().item())
        print("Avg hard negative:", neg_mean.mean().item())

    return loss, pos.mean().item(), neg_mean.mean().item()



class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(
            query=query,
            positive_key=positive_key,
            negative_keys=negative_keys,
            temperature=self.temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode
        )


# -------------------------------------------
# ORIGINAL INFO-NCE with debug
# -------------------------------------------
def info_nce(query, positive_key, negative_keys=None,
                      temperature=0.1, reduction='mean',
                      negative_mode='unpaired', debug=False):

    # normalize
    query = F.normalize(query, dim=1)
    positive_key = F.normalize(positive_key, dim=1)
    if negative_keys is not None:
        negative_keys = F.normalize(negative_keys, dim=-1)

    # ----------------------------------------
    # CASE 1: EXPLICIT NEGATIVES
    # ----------------------------------------
    if negative_keys is not None:

        # positive similarity for each sample
        pos = torch.sum(query * positive_key, dim=1, keepdim=True)

        if debug:
            print("\n[DEBUG] positive logits shape:", pos.shape)
            print(pos)

        # negative similarities
        if negative_mode == 'unpaired':  # neg_keys: (M, D)
            neg = query @ negative_keys.T      # (N, M)

        elif negative_mode == 'paired':       # neg_keys: (N, M, D)
            neg = torch.sum(query.unsqueeze(1) * negative_keys, dim=-1)

        if debug:
            print("\n[DEBUG] negative logits shape:", neg.shape)
            print(neg)

        # final logits
        logits = torch.cat([pos, neg], dim=1)

        # label ALWAYS 0 (positive is first)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

        if debug:
            print("\n[DEBUG] combined logits:")
            print(logits)
            print("[DEBUG] labels:", labels)

    # ----------------------------------------
    # CASE 2: IMPLICIT NEGATIVES (batch negatives)
    # ----------------------------------------
    else:
        logits = query @ positive_key.T  # N × N matrix

        # diagonal entries are positives → labels = [0,1,2,...]
        labels = torch.arange(query.size(0), device=query.device)

        if debug:
            print("\n[DEBUG] N×N logits:")
            print(logits)
            print("[DEBUG] labels:", labels)
            print("Correct positive = diagonal")

    # apply temperature
    logits = logits / temperature
    if debug:
        print("\n[DEBUG] logits / temperature:")
        print(logits)

    loss = F.cross_entropy(logits, labels, reduction=reduction)

    # Track positive/negative means for logging
    if negative_keys is None:
        pos_mean = logits.diag().mean().item()
        neg_mean = (logits.sum(-1) - logits.diag()) / (logits.size(0)-1)
        neg_mean = neg_mean.mean().item()
    else:
        pos_mean = pos.mean().item()
        neg_mean = neg.mean().item()

    return loss, pos_mean, neg_mean



def symmetric_margin_loss(S, T, margin=0.5, k=32):
    S = F.normalize(S, dim=1)
    T = F.normalize(T, dim=1)

    N = S.size(0)

    sim = torch.matmul(S, T.t())  # cosine sim

    # Mask diagonal (positive pairs)
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



