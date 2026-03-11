import os, json
import random
from pathlib import Path
from pyexpat import model
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import torch
import torch.nn.functional as F
device = torch.device("cpu") 
import pickle
import sys, os
sys.path.append('/scratch/hpc-prf-whale/duygu/alignment/Entity_embedding_alignment/NAAS')
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


from modules.models.fusion import (
    AdaptiveFusionMLP, 
    belief_fusion_loss_pure, 
    belief_fusion_loss_hybrid,
    normalize_feature
)
from modules.models.fine_tune import initialize_models_and_update_embeddings
from modules.alignment.agent_alignment import run_two_agent_alignment
from modules.models.base import *
from modules.graph.features import _feature_sims_cross
from modules.graph.build import build_graph_indexes_GCN, build_merged_graph_with_alignment_edges, build_merged_graph
import modules.models.fusion as fusion
fusion.USE_DST_ONLY = False
fusion.USE_NEURO_SYMBOLIC = True
from modules.models.fine_tune import fine_tune_kvsall
import types
import modules.models.train as train_mod
#train_mod.args = types.SimpleNamespace(test_triples=None)
#args = types.SimpleNamespace(test_triples=None)


# === Data loader ===
from modules.data_loader import (
     normalize_embedding_space,load_triples, load_triples_from_files,load_parquet_triples
)

# --- Model imports ---
from modules.models.base import SharedSpaceAlignmentNN, CriticAlignmentNN

# --- Evaluation imports ---
from modules.eval.Entity_alignment import test

# --- Graph utilities ---
from modules.graph.build import build_graph_indexes

def validation_alignment_loss(S_val_aligned, T_val_aligned):
    S_norm = F.normalize(S_val_aligned, p=2, dim=1)
    T_norm = F.normalize(T_val_aligned, p=2, dim=1)
    sim = (S_norm * T_norm).sum(dim=1)
    return (1 - sim).mean()

def debug_raw_similarity(S, T):
    S20 = S[:20]
    T20 = T[:20]

    sim = F.cosine_similarity(S20.unsqueeze(1), T20.unsqueeze(0), dim=2)

    print("\n========== RAW COSINE SIMILARITY CHECK ==========")
    print("Matrix S[:20] × T[:20]:")
    print(sim)

    for i in range(3):  # inspect first 3 rows
        vals, idx = torch.sort(sim[i], descending=True)
        print(f"\nRow {i} sorted sims:")
        print("similarities:", vals[:10])
        print("indices:     ", idx[:10])
        
        
def orthogonal_regularizer(model, weight=1e-3):
    ortho_loss = 0.0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight
            WT_W = W.t() @ W
            I = torch.eye(WT_W.size(0), device=W.device)
            ortho_loss += ((WT_W - I)**2).mean()
    return weight * ortho_loss


def load_oenea_triples(path):
    triples = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples




def train_alignment_model(
    S_train, T_train, input_dim, hidden_dim, epochs, lr, w1, w2, w3, w4,
    entity_embeddings1, entity_embeddings2, relation_embeddings,
    output_dir, triples_batch, kg_1,
    kg_2,device,
    merged_embeddings_normalized_no_target, merged_embeddings_with_target,
    cleaned_alignment_dict, S_train_keys, T_train_keys,S_val_keys, T_val_keys,
    val_triples, train_triples, 
    train_triples_1=None, train_triples_2=None,
    directory_1=None, G1=None, G2=None, S_val=None, T_val=None,    
    S_test_keys=None, T_test_keys=None,use_agents=False, test_triples_path=None
):
    
    #### --------------------------------------------------------
    #### 1) Build EN + FR graphs separately
    #### --------------------------------------------------------
    

    #kg1_triples = load_oenea_triples("/scratch/hpc-prf-whale/duygu/alignment/data/OpenEA_dataset_v1.1/EN_FR_15K_V1/rel_triples_1")
   # kg2_triples = load_oenea_triples("/scratch/hpc-prf-whale/duygu/alignment/data/OpenEA_dataset_v1.1/EN_FR_15K_V1/rel_triples_2")


    #### --------------------------------------------------------
    #### 2) Models: Dual RGCN + Fusion + Proposer
    #### --------------------------------------------------------
    #G = build_merged_graph_with_alignment_edges(
    #kg1_triples, kg2_triples,
    #train_links=list(zip(S_train_keys, T_train_keys))
    #)

    G = build_merged_graph(kg_1, kg_2)
    # --- sanity check alignment edges ---
    print("Relations:", list(G.relation2id.keys())[:20], "...")  # quick peek

    sameas_id = None
    for key in ["sameAs", "owl:sameAs", "SAMEAS", "alignment", "align"]:
        if key in G.relation2id:
            sameas_id = G.relation2id[key]
            sameas_name = key
            break

    print("sameAs relation name:", sameas_name if sameas_id is not None else None)
    print("sameAs id:", sameas_id)

    if sameas_id is None:
        print("[WARN] No sameAs relation found in relation2id. Check builder naming.")
    else:
        print("num sameAs edges:", len(G.adj_lists.get(sameas_id, [])))
        print("train links:", len(S_train_keys))
        # if edges are added in both directions, expect ~2 * train_links
        
   


    rgcn = RGCNEncoder(
        in_dim=input_dim,
        hid_dim=input_dim,
        num_relations=len(G.relation2id)
        )
        
    id2ent = {i: e for e, i in G.entity2id.items()}
    
     # count cross edges (should be 0)
    cross = 0
    for r, edges in G.adj_lists.items():
        for (dst, src) in edges:
            dst_ent = id2ent[dst]
            src_ent = id2ent[src]
            if ("fr.dbpedia.org" in dst_ent) != ("fr.dbpedia.org" in src_ent):
                cross += 1
    print("Cross-KG edges:", cross)

    
    print("\n=== DEBUG: First 3 entries from entity_embeddings1 (EN) ===")
    for i, ent in enumerate(entity_embeddings1.index[:3]):
        print(i, ent)
        print(entity_embeddings1.loc[ent].values[:10], "\n")  # print first 10 dims

    print("\n=== DEBUG: First 3 entries from entity_embeddings2 (FR) ===")
    for i, ent in enumerate(entity_embeddings2.index[:3]):
        print(i, ent)
        print(entity_embeddings2.loc[ent].values[:10], "\n")


    E_list = []
    for i in range(len(id2ent)):
        ent = id2ent[i]
        if ent in entity_embeddings1.index:
            E_list.append(entity_embeddings1.loc[ent].values)
        else:
            E_list.append(entity_embeddings2.loc[ent].values)

    E = torch.tensor(E_list)
    
    print("\n=== DEBUG: First 50 entities in merged graph ===")
    for i, (ent, idx) in enumerate(G.entity2id.items()):
        print(idx, ent)
        if i >= 49:
            break
        
    print("\n=== DEBUG: Last 50 entities in merged graph ===")
    total = len(G.entity2id)

    for i in range(total - 50, total):
        ent = list(G.entity2id.keys())[i]
        print(i, ent)
        
    en_count = 0
    fr_count = 0
    other_count = 0

    for ent in G.entity2id.keys():
        if "dbpedia.org/resource" in ent and not ent.startswith("http://fr."):
            en_count += 1
        elif "fr.dbpedia.org/resource" in ent:
            fr_count += 1
        else:
            other_count += 1

    print("\n=== ENTITY LANGUAGE DISTRIBUTION ===")
    print("EN entities:", en_count)
    print("FR entities:", fr_count)
    print("Other:", other_count)
    
    print("\n=== DEBUG: Check FR embedding overlap ===")
    fr_emb_uris = set(entity_embeddings2.index)
    graph_uris = set(G.entity2id.keys())

    overlap = fr_emb_uris & graph_uris

    print("FR embedding count:", len(fr_emb_uris))
    print("Graph entity count:", len(graph_uris))
    print("FR ∩ Graph:", len(overlap))
    print("Examples:", list(overlap)[:10])


    idx_map = G.entity2id

    S_idx = torch.tensor([idx_map[u] for u in S_train_keys]).long()
    T_idx = torch.tensor([idx_map[v] for v in T_train_keys]).long()

    S_val_idx = torch.tensor([idx_map[u] for u in S_val_keys]).long()
    T_val_idx = torch.tensor([idx_map[v] for v in T_val_keys]).long()


    fusion = Fusion(input_dim)
    proposer = SharedSpaceAlignmentNN(input_dim, hidden_dim)

    optimizer = torch.optim.Adam(
        list(rgcn.parameters()) +
        list(fusion.parameters()) +
        list(proposer.parameters()),
        lr=lr
    )

    best_val_loss = float("inf")
    best_val_mrr = -1.0  
    train_logs = []
    val_logs = []

    #### 4) Training
    #### --------------------------------------------------------
    for epoch in range(epochs):

        # 2) shuffle training link indices
        batch_size = 512
        perm = torch.randperm(len(S_idx), device=S_idx.device)

        total_loss = 0.0
        sum_pos = 0.0
        sum_neg = 0.0
        num_batches = 0

        # -----------------------------
        # TRAIN (mini-batches)
        # -----------------------------
        for bstart in range(0, len(S_idx), batch_size):
            b = perm[bstart:bstart + batch_size]
            Sb = S_idx[b]
            Tb = T_idx[b]
            
            T_all_idx = torch.tensor(
            [idx_map[u] for u in entity_embeddings2.index],  # all FR URIs
            dtype=torch.long, device=E.device
            )


            # forward (recompute H per batch; slower but safe)
            H = rgcn(E, G.adj_lists)
            
            Hs = H[Sb]
            Ht = H[Tb]

            S_fused = fusion(E[Sb], Hs)
            T_fused = fusion(E[Tb], Ht)

            S_align = proposer(S_fused)
            T_align = proposer(T_fused)

            # optional: batch-local keys for debug printing
            #S_keys_b = [S_train_keys[i] for i in b.tolist()]
            #T_keys_b = [T_train_keys[i] for i in b.tolist()]

            # main loss
            loss, pos, neg = symmetric_margin_loss(
                S_align,
                T_align,
                margin=0.8,
                k=32,
                #debug=(epoch == 0 and bstart == 0),  # print once
                #S_keys=S_keys_b,
                #T_keys=T_keys_b
            )

            # variance regularizer (anti-collapse)
            #var_pen = var_loss(S_align) + var_loss(T_align)
            #loss = loss + 1e-3 * var_pen

            # gate regularization (keep alpha near 0.5 early)
            #gate_reg_weight = 5e-2  # try 0.05 first

            xS = torch.cat([E[Sb], Hs], dim=-1)
            xT = torch.cat([E[Tb], Ht], dim=-1)

            aS = torch.sigmoid(fusion.gate(xS))
            aT = torch.sigmoid(fusion.gate(xT))

            def gate_penalty(a, low=0.2, high=0.8):
                return (
                    F.relu(low - a).pow(2).mean() +
                    F.relu(a - high).pow(2).mean()
                )

            gate_reg = 0.1 * (gate_penalty(aS) + gate_penalty(aT))

            loss = loss + gate_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate epoch stats
            total_loss += loss.item()
            sum_pos += float(pos)
            sum_neg += float(neg)
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_pos  = sum_pos / max(1, num_batches)
        avg_neg  = sum_neg / max(1, num_batches)
        print(f"[TRAIN {epoch}] avg_loss={avg_loss:.4f} avg_pos={avg_pos:.4f} avg_neg={avg_neg:.4f}")
        
        with torch.no_grad():

            # compute fusion gate statistics on full train pairs
            H_tmp = rgcn(E, G.adj_lists)

            xS_all = torch.cat([E[S_idx], H_tmp[S_idx]], dim=-1)
            xT_all = torch.cat([E[T_idx], H_tmp[T_idx]], dim=-1)

            aS_all = torch.sigmoid(fusion.gate(xS_all))
            aT_all = torch.sigmoid(fusion.gate(xT_all))

            a_all = torch.cat([aS_all, aT_all], dim=0)

            alpha_struct = a_all.mean().item()
            alpha_base = (1 - a_all).mean().item()

        print(
            f"[FUSION {epoch}] structural={alpha_struct:.4f} base={alpha_base:.4f}"
)

        # -----------------------------
        # PROBE + TRAIN EVAL (no_grad)
        # -----------------------------
        with torch.no_grad():

            # ---- quick collapse probe on a random batch ----
            b_probe = torch.randperm(len(S_idx), device=S_idx.device)[:512]
            Sb_p = S_idx[b_probe]
            Tb_p = T_idx[b_probe]

            H_tmp = rgcn(E, G.adj_lists)
            Hs_tmp = H_tmp[Sb_p]
            Ht_tmp = H_tmp[Tb_p]

            S_tmp = proposer(fusion(E[Sb_p], Hs_tmp))
            T_tmp = proposer(fusion(E[Tb_p], Ht_tmp))

            S_ = F.normalize(S_tmp, dim=1)
            T_ = F.normalize(T_tmp, dim=1)
            sim = S_ @ T_.T

            pos_mean = sim.diag().mean().item()
            off_mean = ((sim.sum() - sim.diag().sum()) / (sim.numel() - sim.size(0))).item()
            print(f"[PROBE {epoch}] pos_mean={pos_mean:.4f} offdiag_mean={off_mean:.4f}")

            # ---- gate stats for SAME sampled batch ----
            #xS_tmp = torch.cat([E[Sb_p], Hs_tmp], dim=-1)
            #aS_tmp = torch.sigmoid(fusion.gate(xS_tmp))
            #print("TRAIN alpha stats (sampled batch):")
            #print("  per-dimension mean (first 10):", aS_tmp.mean(dim=0)[:10].cpu().numpy())
            #print("  per-entity mean   (first 10):", aS_tmp.mean(dim=1)[:10].cpu().numpy())
            #print("  global min / max:", aS_tmp.min().item(), aS_tmp.max().item())

            # ---- full train embeddings for EA evaluation ----
            H_train = rgcn(E, G.adj_lists)
            Hs_t = H_train[S_idx]
            Ht_t = H_train[T_idx]

            S_train_f = fusion(E[S_idx], Hs_t)
            T_train_f = fusion(E[T_idx], Ht_t)
            
            # -------- Fusion analysis --------
            Z = S_train_f

            cos_ZE = F.cosine_similarity(Z, E[S_idx], dim=1).mean().item()
            cos_ZH = F.cosine_similarity(Z, Hs_t, dim=1).mean().item()

            print(f"[FUSION SIM {epoch}] cos(Z,E)={cos_ZE:.4f} cos(Z,H)={cos_ZH:.4f}")


            S_train_a = proposer(S_train_f)
            T_train_a = proposer(T_train_f)

        train_result, train_hits, train_mr, train_mrr = test(
            embeds1=S_train_a.cpu().numpy(),
            embeds2=T_train_a.cpu().numpy(),
            mapping=None,
            top_k=[1, 5, 10, 50],
            threads_num=1,
            metric='cosine',
            normalize=True
        )

        print(
            f"[TRAIN {epoch}] loss={loss.item():.4f} "
            f"pos={pos:.4f} neg={neg:.4f} "
            f"H@1={train_hits[0]:.4f} H@10={train_hits[2]:.4f} "
            f"MRR={train_mrr:.4f} "
            f"struct={alpha_struct:.3f} base={alpha_base:.3f}"
        )

        
        # save train log
        train_logs.append({
            "epoch": epoch,
            "loss": loss.item(),
            "pos": pos,
            "neg": neg,
            "hits1": train_hits[0],
            "hits5": train_hits[1],
            "hits10": train_hits[2],
            "hits50": train_hits[3],
            "mr": train_mr,
            "mrr": train_mrr,
            "alpha_structural": alpha_struct,
            "alpha_base": alpha_base,
            "cos_Z_E": cos_ZE,
            "cos_Z_H": cos_ZH
        })
        #### --------- VALIDATION ---------
        if epoch % 1 == 0:
            with torch.no_grad():

                H_val = rgcn(E, G.adj_lists)

                Hs_val = H_val[S_val_idx]
                Ht_val = H_val[T_val_idx]

                S_val_f = fusion(E[S_val_idx], Hs_val)
                T_val_f = fusion(E[T_val_idx], Ht_val)

                S_val_a = proposer(S_val_f)
                T_val_a = proposer(T_val_f)

                val_loss, val_pos, val_neg = symmetric_margin_loss(S_val_a, T_val_a)
                
                #val_loss, val_pos, val_neg = symmetric_margin_loss(S_val_a, T_val_a, margin=0.8, k=32)


                # ---- Compute ranking metrics on VAL ----
                S_val_np = S_val_a.cpu().numpy()
                T_val_np = T_val_a.cpu().numpy()

                val_result, val_hits, val_mr, val_mrr = test(
                    embeds1=S_val_np,
                    embeds2=T_val_np,
                    mapping=None,
                    top_k=[1, 5, 10, 50],
                    threads_num=1,
                    metric='cosine',
                    normalize=True
                )

                print(f"[VAL   {epoch}] loss={val_loss:.4f} pos={val_pos:.4f} neg={val_neg:.4f} "
                    f"H@1={val_hits[0]:.4f}  H@10={val_hits[2]:.4f}  MRR={val_mrr:.4f}")

                # save val metrics
                val_logs.append({
                    "epoch": epoch,
                    "loss": val_loss.item(),
                    "pos": val_pos,
                    "neg": val_neg,
                    "hits1": val_hits[0],
                    "hits5": val_hits[1],
                    "hits10": val_hits[2],
                    "hits50": val_hits[3],
                    "mr": val_mr,
                    "mrr": val_mrr
                })


            # === Save best model based on VAL loss ===

                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    torch.save(proposer.state_dict(), os.path.join(output_dir, "best_val_proposer.pt"))
                    torch.save(fusion.state_dict(),    os.path.join(output_dir, "best_val_fusion.pt"))
                    torch.save(rgcn.state_dict(),      os.path.join(output_dir, "best_val_rgcn.pt"))
                    print(f"Saved BEST model at epoch {epoch} (val MRR={val_mrr:.4f})")
                #S_hat, S_shared = proposer(S_train_tensor)
                #T_hat, T_shared = proposer(T_train_tensor)
            
            
    pd.DataFrame(train_logs).to_csv(os.path.join(output_dir, "train_metrics.csv"), index=False)
    pd.DataFrame(val_logs).to_csv(os.path.join(output_dir, "val_metrics.csv"), index=False)

    print("Saved training and validation metrics to CSV.")


            # --- Simple alignment loss ---
           # loss, pos_mean, neg_mean = whiteboard_loss(S_hat, T_hat , margin=2.0)

            #loss.backward()
            #optimizer.step()
            
            #if (epoch+1) % 5 == 0:
               # print(f"[Epoch {epoch+1}] loss={loss.item():.4f} pos={pos_mean:.4f} neg={neg_mean:.4f}")

    # --------------------------------------------
    # 1. Load saved modules
    # --------------------------------------------
    proposer = SharedSpaceAlignmentNN(input_dim, hidden_dim)
    fusion   = Fusion(input_dim)
    rgcn     = RGCNEncoder(input_dim, input_dim, num_relations=len(G.relation2id))

    proposer.load_state_dict(torch.load(os.path.join(output_dir, "best_val_proposer.pt")))
    fusion.load_state_dict(torch.load(os.path.join(output_dir, "best_val_fusion.pt")))
    rgcn.load_state_dict(torch.load(os.path.join(output_dir, "best_val_rgcn.pt")))

    proposer.eval()
    fusion.eval()
    rgcn.eval()

    # --------------------------------------------
    # 2. Prepare S_test and T_test embeddings
    # (use SAME merged graph ordering)
    # --------------------------------------------
    S_test_idx = torch.tensor([G.entity2id[u] for u in S_test_keys]).long()
    T_test_idx = torch.tensor([G.entity2id[v] for v in T_test_keys]).long()

    E_test = E  # global normalized base embeddings used during training

    with torch.no_grad():

        # 3. Compute structural embeddings
        H_test = rgcn(E_test, G.adj_lists)

        Hs_test = H_test[S_test_idx]
        Ht_test = H_test[T_test_idx]

        # 4. Fuse base + structure
        S_test_fused = fusion(E_test[S_test_idx], Hs_test)
        T_test_fused = fusion(E_test[T_test_idx], Ht_test)

        # 5. Shared space proposer
        S_test_aligned = proposer(S_test_fused)
        T_test_aligned = proposer(T_test_fused)
        
   # --------------------------------------------
    # Build aligned embeddings for ALL entities
    # --------------------------------------------
    with torch.no_grad():

        # Structural embeddings
        H_all = rgcn(E, G.adj_lists)

        # Fuse base + structure
        fused_all = fusion(E, H_all)

        # Project to shared space
        aligned_all = proposer(fused_all)

    # Convert to numpy
    aligned_entities_all = aligned_all.cpu().numpy()
    
    # Convert to numpy
    aligned_entities_all = aligned_all.cpu().numpy()

    # Recover URI order
    entity_order = [id2ent[i] for i in range(len(id2ent))]

    # Attach URIs
    aligned_entities_all = aligned_entities_all[:len(entity_order)]
    aligned_df = pd.DataFrame(aligned_entities_all, index=entity_order)
    print("aligned embeddings:", aligned_entities_all.shape)
    print("entity_order:", len(entity_order))

    # Remove FR entities that are aligned
    aligned_df_no_fr = aligned_df.drop(index=T_train_keys, errors="ignore")

    aligned_entities_filtered = aligned_df_no_fr.values

    # Convert to numpy
    S_eval = S_test_aligned.cpu().numpy()
    T_eval = T_test_aligned.cpu().numpy()

    # --------------------------------------------
    # 6. Evaluate entity alignment
    # --------------------------------------------
    #alignment_result, hits_at_k, mean_rank, mrr = test(
        #embeds1=S_eval,
        #embeds2=T_eval,
        #mapping=None,
        #top_k=[1, 5, 10, 50],
        #threads_num=1,
        #metric='cosine',
        #normalize=True
    #)

    #print(f"\n[Final Test Alignment Results]")
    #print(f"Pairs used: {len(S_test_keys)}")
    #print(f"Hits@1:  {hits_at_k[0]:.4f}")
    #print(f"Hits@5:  {hits_at_k[1]:.4f}")
    #print(f"Hits@10: {hits_at_k[2]:.4f}")
    #print(f"Hits@50: {hits_at_k[3]:.4f}")
    #print(f"MR:      {mean_rank:.2f}")
    #print(f"MRR:     {mrr:.4f}")
    
    # Save test metrics
    #test_results = {
        #"pairs_used": len(S_test_keys),
        #"hits1": hits_at_k[0],
        #"hits5": hits_at_k[1],
        #"hits10": hits_at_k[2],
        #"hits50": hits_at_k[3],
        #"mr": mean_rank,
        #"mrr": mrr
    #}

    # Save as JSON
    #with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        #json.dump(test_results, f, indent=4)

    # Also save as CSV (optional but convenient)
    #pd.DataFrame([test_results]).to_csv(
        #os.path.join(output_dir, "test_metrics.csv"),
        #index=False
    #)

    #print(f"Saved test metrics to {output_dir}")


    # ===== fine-tuning =====
    model = initialize_models_and_update_embeddings(
        aligned_entities_filtered,
        relation_embeddings.values,
        output_dir, directory_1=directory_1
    )

    print(f" Shape of aligned_entities_all: {aligned_entities_all.shape}\n")
    entity_to_idx_aligned = {uri: idx for idx, uri in enumerate(aligned_df_no_fr.index)}
    aligned_relation_embeddings = relation_embeddings.values
    relation_to_idx_aligned = {uri: idx for idx, uri in enumerate(relation_embeddings.index)}
    #torch.save(model.state_dict(), os.path.join(output_dir, "pre_finetune.pt"))
    fine_tune_folder = os.path.join(output_dir, "fine_tune")
    os.makedirs(fine_tune_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(fine_tune_folder, "model.pt"))
    
    print(f"Saved aligned model to {fine_tune_folder}")


     # === Fine-tuning and final model save ===
    try:
        # If running through pipeline.py → args.test_triples will exist
        #test_triples_path = args.test_triples_path if 'args' in globals() else None
        test_triples_path = test_triples_path
    except NameError:
        test_triples_path = None

    # Use only a small subset of triples for testing
    import random
    random.shuffle(triples_batch)
    triples_batch_small = triples_batch[:60000]
    # Now proceed based on availability
    if test_triples_path and os.path.exists(test_triples_path):
        print("\n[Test triples found — proceeding with fine-tuning and link prediction evaluation]")
        try:
            fine_tune_kvsall(
                model=model,
                aligned_entity_embeddings=aligned_entities_filtered,
                aligned_relation_embeddings=aligned_relation_embeddings                                                         ,
                triples_batch=triples_batch_small,
                val_triples=val_triples,
                train_triples=train_triples,
                entity_to_idx=entity_to_idx_aligned,
                relation_to_idx=relation_to_idx_aligned,
                device=device,
                output_dir=output_dir,
                batch_size=256,
                epochs=20,
                lr=0.001
            )
        except NameError:
            print("fine_tune_kvsall not defined — skipping fine-tuning step.")
    else:
        print("\n[Note] No test triples provided — skipping fine-tuning and link prediction.]")
        
         # Build final reduced model (drop target-side entities)
        # -------------------------------------------------
    # Build final model after fine-tuning
    # -------------------------------------------------

    fine_tuned_entities = model.entity_embeddings.weight.data.cpu().numpy()
    fine_tuned_relations = model.relation_embeddings.weight.data.cpu().numpy()

    # Reattach URIs using the same order used during training
    fine_tuned_df = pd.DataFrame(
        fine_tuned_entities,
        index=aligned_df_no_fr.index
    )

    final_aligned_df = fine_tuned_df
    final_aligned_entities = final_aligned_df.values

    final_model_folder = os.path.join(output_dir, "final_aligned_model")
    os.makedirs(final_model_folder, exist_ok=True)

    entity_to_idx_final = {
        uri: idx for idx, uri in enumerate(final_aligned_df.index)
    }

    relation_to_idx_final = {
        uri: idx for idx, uri in enumerate(relation_embeddings.index)
    }
    

    print("\n=== ENTITY INDEX CHECK ===")

    sample_uris = random.sample(list(entity_to_idx_final.keys()), 20)

    for uri in sample_uris:
        idx = entity_to_idx_final[uri]

        emb_from_df = final_aligned_df.loc[uri].values
        emb_from_matrix = final_aligned_entities[idx]

        diff = ((emb_from_df - emb_from_matrix) ** 2).sum()

        print(uri)
        print("idx:", idx, "diff:", diff)
        
        
    print("\n=== RELATION INDEX CHECK ===")

    for uri, idx in list(relation_to_idx_final.items())[:10]:
        emb_from_matrix = fine_tuned_relations[idx]

        print(uri, "-> index:", idx)
        print("embedding first dims:", emb_from_matrix[:5])
            
    print("\n=== SHAPE CHECK ===")

    print("Entity embeddings:", final_aligned_entities.shape)
    print("Relation embeddings:", fine_tuned_relations.shape)

    print("Entity dict size:", len(entity_to_idx_final))
    print("Relation dict size:", len(relation_to_idx_final))

    final_model = initialize_models_and_update_embeddings(
        final_aligned_entities,
        fine_tuned_relations,
        final_model_folder,
        directory_1=directory_1
    )

    torch.save(
        final_model.state_dict(),
        os.path.join(final_model_folder, "model.pt")
    )

    print(f"Final model saved in {final_model_folder}")

    return final_model