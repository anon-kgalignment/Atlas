import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .data.graph import KGGraph, get_pair_indices_separate
from .models.gat import SiameseRelationalGATEncoder
from .models.fusion import AdaptiveFusion
from .models.projector import DualProjector
from .models.loss import infonce_loss, infonce_global_loss, MIDiscriminator, mi_jsd_loss


# ─────────────────────────────────────────────
# CROSS-KG NEAREST-NEIGHBOR REFINEMENT
# ─────────────────────────────────────────────

def _knn_context(
    Q          : torch.Tensor,
    K          : torch.Tensor,
    k          : int   = 10,
    temperature: float = 0.07,
    chunk_size : int   = 2000,
) -> torch.Tensor:
    
    N, D = Q.size()
    context = torch.zeros(N, D, device=Q.device, dtype=Q.dtype)

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        sim   = Q[start:end] @ K.T                    # [chunk, N_K]
        topk_vals, topk_idx = sim.topk(k, dim=1)      # [chunk, k]
        attn  = F.softmax(topk_vals / temperature, dim=1)  # [chunk, k]
        K_sel = K[topk_idx]                            # [chunk, k, D]
        context[start:end] = (attn.unsqueeze(-1) * K_sel).sum(dim=1)

    return context


@torch.no_grad()
def cross_knn_refine(
    A1         : torch.Tensor,
    A2         : torch.Tensor,
    k          : int   = 10,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    A1 = F.normalize(A1, dim=1)
    A2 = F.normalize(A2, dim=1)

    ctx1 = _knn_context(A1, A2, k, temperature)
    A1_r = F.normalize(A1 + ctx1, dim=1)

    ctx2 = _knn_context(A2, A1, k, temperature)
    A2_r = F.normalize(A2 + ctx2, dim=1)

    return A1_r, A2_r


# ─────────────────────────────────────────────
# EVALUATION FUNCTION
# ─────────────────────────────────────────────


def evaluate_alignment(
    A1     : torch.Tensor,
    A2     : torch.Tensor,
    src_ids,
    tgt_ids,
    top_k  : List[int] = [1, 5, 10, 50],
) -> Dict:
    """
    Evaluate entity alignment.
    A1: KG1 aligned embeddings [N1, dim]
    A2: KG2 aligned embeddings [N2, dim]
    src_ids: KG1-local indices of test pairs
    tgt_ids: KG2-local indices of test pairs
    """
    from modules.eval.eval import test

    A_src = A1[src_ids].cpu().numpy()
    A_tgt = A2[tgt_ids].cpu().numpy()

    _, hits, mr, mrr = test(
        embeds1    = A_src,
        embeds2    = A_tgt,
        mapping    = None,
        top_k      = top_k,
        threads_num= 4,
        metric     = 'cosine',
        normalize  = True,
        accurate   = True,
    )

    results = {}
    for i, k in enumerate(top_k):
        results[f"Hits@{k}"] = hits[i]
    results["MRR"] = mrr
    results["MR"]  = mr
    return results


# ─────────────────────────────────────────────
# BUILD PER-KG RELATION EMBEDDING TENSOR
# ─────────────────────────────────────────────

def build_rel_emb_tensor(
    G       : KGGraph,
    rel_emb : pd.DataFrame,
    dim     : int,
    device  : torch.device,
) -> torch.Tensor:
    """Build relation embedding tensor indexed by G.relation2id."""
    R = torch.zeros(G.n_relations, dim, device=device, dtype=torch.float32)
    for r_uri, r_id in G.relation2id.items():
        if r_uri in rel_emb.index:
            R[r_id] = torch.tensor(
                rel_emb.loc[r_uri].values, dtype=torch.float32
            ).to(device)
    return R


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────

def train_atlas(
    data            : Dict,
    G1              : KGGraph,
    G2              : KGGraph,
    E1              : np.ndarray,
    E2              : np.ndarray,
    P1              : np.ndarray,
    P2              : np.ndarray,
    output_dir      : str,
    # Model hyperparameters
    hidden_dim      : int   = 256,
    labse_dim       : int   = 1024,
    n_heads         : int   = 4,
    dropout         : float = 0.1,
    # Training hyperparameters
    epochs          : int   = 10,
    batch_size      : int   = 512,
    lr              : float = 1e-3,
    # Loss hyperparameters
    temperature     : float = 0.07,
    lambda_div      : float = 0.01,
    lambda_mi       : float = 0.0,
    # Evaluation
    eval_every      : int   = 1,
    patience        : int   = 20,
    # Device
    device_str      : str   = "auto",
) -> Dict:
    """
    Train SAGE with siamese GAT encoder (shared weights for both KGs).

    Args:
        data   : output of load_all()
        G1, G2 : separate KGGraph objects (no ALIGN edges)
        E1, E2 : DICE embedding matrices [N1/N2, dim]
        P1, P2 : LaBSE embedding matrices [N1/N2, 768]
    """
    os.makedirs(output_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"\n{'='*55}")
    print(f"  SAGE TRAINING (siamese GAT encoder)")
    print(f"{'='*55}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  LR           : {lr}")
    print(f"  Temperature  : {temperature}")
    print(f"  Lambda Div   : {lambda_div}")
    print(f"  Hidden dim   : {hidden_dim}")
    print(f"  N heads      : {n_heads}")
    print(f"{'='*55}\n")

    E1_t = torch.tensor(E1, dtype=torch.float32).to(device)
    E2_t = torch.tensor(E2, dtype=torch.float32).to(device)
    P1_t = torch.tensor(P1, dtype=torch.float32).to(device)
    P2_t = torch.tensor(P2, dtype=torch.float32).to(device)

    encoder = SiameseRelationalGATEncoder(
        dice_dim        = data["dim"],
        labse_dim       = labse_dim,
        hidden_dim      = hidden_dim,
        n_relations_kg1 = G1.n_relations,
        n_relations_kg2 = G2.n_relations,
        n_heads         = n_heads,
        dropout         = dropout,
    ).to(device)

    fusion = AdaptiveFusion(
        dice_dim   = data["dim"],
        labse_dim  = labse_dim,
        hidden_dim = hidden_dim,
        dropout    = dropout,
    ).to(device)

    projector = DualProjector(
        hidden_dim = hidden_dim,
        dropout    = dropout,
    ).to(device)

    discriminator = MIDiscriminator(
        hidden_dim = hidden_dim,
        dropout    = dropout,
    ).to(device) if lambda_mi > 0.0 else None

    # ── Optimizer ────────────────────────────────
    all_params = (
        list(encoder.parameters()) +
        list(fusion.parameters()) +
        list(projector.parameters())
    )
    if discriminator is not None:
        all_params += list(discriminator.parameters())

    optimizer = torch.optim.Adam(
        all_params,
        lr           = lr,
        weight_decay = 1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    train_src_np, train_tgt_np = get_pair_indices_separate(data["train_pairs"], G1, G2)
    val_src_np,   val_tgt_np   = get_pair_indices_separate(data["val_pairs"],   G1, G2)
    test_src_np,  test_tgt_np  = get_pair_indices_separate(data["test_pairs"],  G1, G2)

    train_src = torch.tensor(train_src_np, dtype=torch.long)
    train_tgt = torch.tensor(train_tgt_np, dtype=torch.long)
    val_src   = torch.tensor(val_src_np,   dtype=torch.long)
    val_tgt   = torch.tensor(val_tgt_np,   dtype=torch.long)
    test_src  = torch.tensor(test_src_np,  dtype=torch.long)
    test_tgt  = torch.tensor(test_tgt_np,  dtype=torch.long)

    best_val_mrr      = -1.0
    best_epoch        = 0
    val_history       = []
    epochs_no_improve = 0
    total_params = (
        sum(p.numel() for p in encoder.parameters()) +
        sum(p.numel() for p in fusion.parameters()) +
        sum(p.numel() for p in projector.parameters())
    )
    print(f"[train] Total parameters: {total_params:,}")
    print(f"[train] Train pairs : {len(train_src)}")
    print(f"[train] Val pairs   : {len(val_src)}")
    print(f"[train] Test pairs  : {len(test_src)}")
    print()

    for epoch in range(epochs):

        encoder.train()
        fusion.train(); projector.train()
        if discriminator is not None:
            discriminator.train()

        perm           = torch.randperm(len(train_src))
        train_src_shuf = train_src[perm]
        train_tgt_shuf = train_tgt[perm]

        epoch_losses = defaultdict(float)
        n_batches    = 0

        for batch_start in range(0, len(train_src_shuf), batch_size):
            batch_end = min(batch_start + batch_size, len(train_src_shuf))
            src_batch = train_src_shuf[batch_start:batch_end]
            tgt_batch = train_tgt_shuf[batch_start:batch_end]

            H1 = encoder.forward_kg1(E1_t, P1_t, G1.adj_lists, device)
            H2 = encoder.forward_kg2(E2_t, P2_t, G2.adj_lists, device)

            Z1, gw1 = fusion(E1_t, P1_t, H1)
            Z2, gw2 = fusion(E2_t, P2_t, H2)

            A1 = projector.forward_kg1(Z1) 
            A2 = projector.forward_kg2(Z2)  

            A_src = A1[src_batch]
            A_tgt = A2[tgt_batch]

            align_loss, pos_sim, neg_sim = infonce_loss(A_src, A_tgt, temperature)

            div_loss = torch.tensor(0.0, device=device)
            if lambda_div > 0:
                gw1_b = gw1[src_batch]
                gw2_b = gw2[tgt_batch]
                ent1  = -(gw1_b * torch.log(gw1_b.clamp(1e-8))).sum(-1).mean()
                ent2  = -(gw2_b * torch.log(gw2_b.clamp(1e-8))).sum(-1).mean()
                div_loss = -(ent1 + ent2) / 2

            mi_loss_val = torch.tensor(0.0, device=device)
            if discriminator is not None:
                Z1_pairs = Z1[src_batch]
                Z2_pairs = Z2[tgt_batch]
                mi_loss_val = mi_jsd_loss(discriminator, Z1_pairs, Z2_pairs)

            total_loss = align_loss + lambda_div * div_loss + lambda_mi * mi_loss_val

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                all_params,
                max_norm=1.0,
            )
            optimizer.step()

            epoch_losses["total_loss"] += total_loss.item()
            epoch_losses["align_loss"] += align_loss.item()
            epoch_losses["div_loss"]   += div_loss.item()
            epoch_losses["mi_loss"]    += mi_loss_val.item()
            epoch_losses["pos_sim"]    += pos_sim
            epoch_losses["neg_sim"]    += neg_sim
            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        scheduler.step()

        if epoch % eval_every == 0:
            encoder.eval()
            fusion.eval(); projector.eval()
            if discriminator is not None:
                discriminator.eval()

            with torch.no_grad():
                H1_v = encoder.forward_kg1(E1_t, P1_t, G1.adj_lists, device)
                H2_v = encoder.forward_kg2(E2_t, P2_t, G2.adj_lists, device)
                Z1_v, gw1_v = fusion(E1_t, P1_t, H1_v)
                Z2_v, gw2_v = fusion(E2_t, P2_t, H2_v)
                A1_v = projector.forward_kg1(Z1_v)
                A2_v = projector.forward_kg2(Z2_v)

            val_results = evaluate_alignment(A1_v, A2_v, val_src, val_tgt)
            val_mrr   = val_results["MRR"]
            val_hits1 = val_results["Hits@1"]

            avg_gw1 = gw1_v.mean(dim=0).cpu().numpy()
            avg_gw2 = gw2_v.mean(dim=0).cpu().numpy()

            print(
                f"[Epoch {epoch:03d}] "
                f"loss={avg_losses['total_loss']:.4f} "
                f"align={avg_losses['align_loss']:.4f} "
                f"div={avg_losses['div_loss']:.4f} "
                f"mi={avg_losses['mi_loss']:.4f} "
                f"| val H@1={val_hits1:.4f} MRR={val_mrr:.4f} "
                f"| gate1=[{avg_gw1[0]:.2f},{avg_gw1[1]:.2f},{avg_gw1[2]:.2f}] "
                f"gate2=[{avg_gw2[0]:.2f},{avg_gw2[1]:.2f},{avg_gw2[2]:.2f}]"
            )

            val_history.append({
                "epoch": epoch, "val_mrr": val_mrr,
                "val_hits1": val_hits1, **avg_losses,
            })

            if val_mrr > best_val_mrr:
                best_val_mrr      = val_mrr
                best_epoch        = epoch
                epochs_no_improve = 0
                ckpt = {
                    "epoch"     : epoch,
                    "val_mrr"   : val_mrr,
                    "encoder"   : encoder.state_dict(),
                    "fusion"    : fusion.state_dict(),
                    "projector" : projector.state_dict(),
                }
                if discriminator is not None:
                    ckpt["discriminator"] = discriminator.state_dict()
                torch.save(ckpt, os.path.join(output_dir, "best_model.pt"))
                print(f"  ✓ New best model saved (val MRR={val_mrr:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  No improvement ({epochs_no_improve}/{patience})")
                if epochs_no_improve >= patience:
                    print(f"\n[train] Early stopping at epoch {epoch}")
                    break

    print(f"\n[train] Loading best model from epoch {best_epoch}")
    checkpoint = torch.load(
        os.path.join(output_dir, "best_model.pt"), map_location=device
    )
    encoder.load_state_dict(checkpoint["encoder"])
    fusion.load_state_dict(checkpoint["fusion"])
    projector.load_state_dict(checkpoint["projector"])
    if discriminator is not None and "discriminator" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator"])

    encoder.eval()
    fusion.eval(); projector.eval()

    with torch.no_grad():
        H1_f = encoder.forward_kg1(E1_t, P1_t, G1.adj_lists, device)
        H2_f = encoder.forward_kg2(E2_t, P2_t, G2.adj_lists, device)
        Z1_f, _ = fusion(E1_t, P1_t, H1_f)
        Z2_f, _ = fusion(E2_t, P2_t, H2_f)
        A1_final = projector.forward_kg1(Z1_f)
        A2_final = projector.forward_kg2(Z2_f)

    # ── Without refinement ───────────────────────
    test_results = evaluate_alignment(A1_final, A2_final, test_src, test_tgt)

    print(f"\n{'='*55}")
    print(f"  FINAL TEST RESULTS (siamese GAT, no refinement)")
    print(f"{'='*55}")
    print(f"  Hits@1  : {test_results['Hits@1']:.4f}")
    print(f"  Hits@5  : {test_results['Hits@5']:.4f}")
    print(f"  Hits@10 : {test_results['Hits@10']:.4f}")
    print(f"  Hits@50 : {test_results['Hits@50']:.4f}")
    print(f"  MRR     : {test_results['MRR']:.4f}")
    print(f"  MR      : {test_results['MR']:.2f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"{'='*55}")

    # ── Cross-KG KNN refinement (test-time, 2 rounds) ──────
    print(f"\n[refine] Applying cross-KG KNN refinement (k=10, 2 rounds)...")
    A1_r, A2_r = cross_knn_refine(A1_final, A2_final, k=10, temperature=0.07)
    A1_r, A2_r = cross_knn_refine(A1_r, A2_r, k=10, temperature=0.07)
    test_results_r = evaluate_alignment(A1_r, A2_r, test_src, test_tgt)

    print(f"\n{'='*55}")
    print(f"  FINAL TEST RESULTS (siamese GAT + KNN 2-round)")
    print(f"{'='*55}")
    print(f"  Hits@1  : {test_results_r['Hits@1']:.4f}")
    print(f"  Hits@5  : {test_results_r['Hits@5']:.4f}")
    print(f"  Hits@10 : {test_results_r['Hits@10']:.4f}")
    print(f"  Hits@50 : {test_results_r['Hits@50']:.4f}")
    print(f"  MRR     : {test_results_r['MRR']:.4f}")
    print(f"  MR      : {test_results_r['MR']:.2f}")
    print(f"  Best epoch: {best_epoch}")
    print(f"{'='*55}")

    # Use the refined result as the final result
    test_results = test_results_r
    A1_final, A2_final = A1_r, A2_r

    results = {
        "test_results" : test_results,
        "best_epoch"   : best_epoch,
        "best_val_mrr" : best_val_mrr,
        "val_history"  : val_history,
        "hyperparams"  : {
            "hidden_dim"  : hidden_dim,
            "n_heads"     : n_heads,
            "epochs"      : epochs,
            "batch_size"  : batch_size,
            "lr"          : lr,
            "temperature" : temperature,
            "lambda_div"  : lambda_div,
        },
        "mode": "siamese_gat",
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    kg1_uris = [G1.id2entity[i] for i in range(G1.n_entities)]
    kg2_uris = [G2.id2entity[i] for i in range(G2.n_entities)]

    A_kg1_df = pd.DataFrame(A1_final.cpu().numpy(), index=kg1_uris)
    A_kg2_df = pd.DataFrame(A2_final.cpu().numpy(), index=kg2_uris)

    A_kg1_df.to_csv(os.path.join(output_dir, "aligned_kg1.csv"))
    A_kg2_df.to_csv(os.path.join(output_dir, "aligned_kg2.csv"))

    print(f"[train] Aligned embeddings saved: "
          f"KG1={A_kg1_df.shape}, KG2={A_kg2_df.shape}")

    # Return encoder twice (as gat1, gat2) for call-site compatibility
    return results, encoder, encoder, fusion, projector, A1_final, A2_final