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
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


from modules.models.fine_tune import initialize_models_and_update_embeddings
from modules.models.base import *
#from modules.graph.features import _feature_sims_cross
from modules.graph.build import build_merged_graph
from modules.models.fine_tune import fine_tune_kvsall
import types
import modules.models.train as train_mod


# === Data loader ===
from modules.data_loader import (
     normalize_embedding_space,load_triples, load_triples_from_files,load_parquet_triples
)

# --- Model imports ---
from modules.models.base import SharedSpaceAlignmentNN

# --- Evaluation imports ---
from modules.eval.Entity_alignment import test


def load_oenea_triples(path):
    triples = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples



def train_alignment_model(
    input_dim, hidden_dim, epochs, lr,
    entity_embeddings1, entity_embeddings2, relation_embeddings,
    output_dir, triples_batch, kg_1,
    kg_2,device,
    #cleaned_alignment_dict, 
    S_train_keys, T_train_keys,S_val_keys, T_val_keys,
    val_triples, train_triples,
    directory_1=None,
    S_test_keys=None, T_test_keys=None, test_triples_path=None
):
    

    G = build_merged_graph(kg_1, kg_2)

    rgcn = RGCNEncoder(
        in_dim=input_dim,
        hid_dim=input_dim,
        num_relations=len(G.relation2id)
        )
        
    id2ent = {i: e for e, i in G.entity2id.items()}


    E_list = []
    for i in range(len(id2ent)):
        ent = id2ent[i]
        if ent in entity_embeddings1.index:
            E_list.append(entity_embeddings1.loc[ent].values)
        else:
            E_list.append(entity_embeddings2.loc[ent].values)

    E = torch.tensor(E_list)

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

            # main loss
            loss, pos, neg = symmetric_margin_loss(
                S_align,
                T_align,
                margin=0.8,
                k=32
            )

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


            # === Save best model based on VAL MRR ===
                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    torch.save(proposer.state_dict(), os.path.join(output_dir, "best_val_proposer.pt"))
                    torch.save(fusion.state_dict(),    os.path.join(output_dir, "best_val_fusion.pt"))
                    torch.save(rgcn.state_dict(),      os.path.join(output_dir, "best_val_rgcn.pt"))
                    print(f"Saved BEST model at epoch {epoch} (val MRR={val_mrr:.4f})")

            
    pd.DataFrame(train_logs).to_csv(os.path.join(output_dir, "train_metrics.csv"), index=False)
    pd.DataFrame(val_logs).to_csv(os.path.join(output_dir, "val_metrics.csv"), index=False)

    print("Saved training and validation metrics to CSV.")


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

    E_test = E  

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
    alignment_result, hits_at_k, mean_rank, mrr = test(
        embeds1=S_eval,
        embeds2=T_eval,
        mapping=None,
        top_k=[1, 5, 10, 50],
        threads_num=1,
        metric='cosine',
        normalize=True
    )

    print(f"\n[Final Test Alignment Results]")
    print(f"Pairs used: {len(S_test_keys)}")
    print(f"Hits@1:  {hits_at_k[0]:.4f}")
    print(f"Hits@5:  {hits_at_k[1]:.4f}")
    print(f"Hits@10: {hits_at_k[2]:.4f}")
    print(f"Hits@50: {hits_at_k[3]:.4f}")
    print(f"MR:      {mean_rank:.2f}")
    print(f"MRR:     {mrr:.4f}")
    
    # Save test metrics
    test_results = {
        "pairs_used": len(S_test_keys),
        "hits1": hits_at_k[0],
        "hits5": hits_at_k[1],
        "hits10": hits_at_k[2],
        "hits50": hits_at_k[3],
        "mr": mean_rank,
        "mrr": mrr
    }

    # Save as JSON
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    # Also save as CSV (optional but convenient)
    pd.DataFrame([test_results]).to_csv(
        os.path.join(output_dir, "test_metrics.csv"),
        index=False
    )

    print(f"Saved test metrics to {output_dir}")


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
    fine_tune_folder = os.path.join(output_dir, "fine_tune")
    os.makedirs(fine_tune_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(fine_tune_folder, "model.pt"))
    
    print(f"Saved aligned model to {fine_tune_folder}")


     # === Fine-tuning and final model save ===
    try:
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