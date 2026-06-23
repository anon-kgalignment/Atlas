# sage/modules/eval/link_prediction.py

import os
import json
import pickle
import shutil
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


# ─────────────────────────────────────────────
# INITIALIZE MODEL WITH ALIGNED EMBEDDINGS
# ─────────────────────────────────────────────

def initialize_model_with_aligned_embeddings(
    entity_embeddings   : np.ndarray,
    relation_embeddings : np.ndarray,
    output_dir          : str,
    directory_kg1       : str,
) -> object:
    """
    Initialize a DICE KGE model with aligned embeddings.
    """
    from dicee import intialize_model

    fine_tune_folder = os.path.join(output_dir, "fine_tune")
    os.makedirs(fine_tune_folder, exist_ok=True)

    for fname in ["configuration.json", "report.json"]:
        src = os.path.join(directory_kg1, fname)
        dst = os.path.join(fine_tune_folder, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            if fname == "configuration.json":
                raise FileNotFoundError(
                    f"configuration.json not found in {directory_kg1}"
                )

    config_path = os.path.join(fine_tune_folder, "configuration.json")
    with open(config_path) as f:
        configs = json.load(f)

    print(f"[lp] Model type: {configs['model']}")
    configs["p"]             = 0
    configs["num_entities"]  = entity_embeddings.shape[0]
    configs["num_relations"] = relation_embeddings.shape[0]
    configs["embedding_dim"] = entity_embeddings.shape[1]

    if configs.get("model") == "Keci":
        configs["q"] = 1

    with open(config_path, "w") as f:
        json.dump(configs, f, indent=2)

    model, _ = intialize_model(configs)

    ent_tensor = torch.tensor(
        entity_embeddings, dtype=torch.float32
    )
    rel_tensor = torch.tensor(
        relation_embeddings, dtype=torch.float32
    )

    with torch.no_grad():
        model.entity_embeddings.weight.data.copy_(
            ent_tensor.contiguous()
        )
        model.relation_embeddings.weight.data.copy_(
            rel_tensor.contiguous()
        )
        model.entity_embeddings.weight.requires_grad  = False
        model.relation_embeddings.weight.requires_grad = False

    print(f"[lp] Model initialized:")
    print(f"     Entities  : {entity_embeddings.shape[0]}")
    print(f"     Relations : {relation_embeddings.shape[0]}")
    print(f"     Dim       : {entity_embeddings.shape[1]}")

    return model


# ─────────────────────────────────────────────
# RECOMPUTE RELATIONS FROM ALIGNED ENTITIES
# ─────────────────────────────────────────────

def _load_raw_triples(path: str) -> List:
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["h", "r", "t"], dtype=str, engine="python",
    ).dropna()
    return df.values.tolist()


def recompute_relation_embeddings(
    rel_emb_np    : np.ndarray,
    r_idx2uri     : Dict,
    triples       : List,
    entity_to_idx : Dict,
    aligned_emb   : np.ndarray,
    kg_name       : str = "",
) -> np.ndarray:
    """
    Re-estimate relation embeddings in the aligned entity space.

    For each relation r: new_r = mean(emb[t] - emb[h]) over all
    triples (h, r, t) where both h and t exist in entity_to_idx.

    This restores TransE geometry (h + r ≈ t) after entity alignment
    has moved entities to a shared space.
    Falls back to original embedding when no triples are found.
    """
    r_uri_to_idx = {uri: i for i, uri in r_idx2uri.items()}

    rel_sum   = np.zeros_like(rel_emb_np, dtype=np.float64)
    rel_count = np.zeros(len(rel_emb_np), dtype=np.int64)

    for h_uri, r_uri, t_uri in triples:
        r_idx = r_uri_to_idx.get(r_uri)
        h_idx = entity_to_idx.get(h_uri)
        t_idx = entity_to_idx.get(t_uri)
        if r_idx is None or h_idx is None or t_idx is None:
            continue
        rel_sum[r_idx]   += aligned_emb[t_idx] - aligned_emb[h_idx]
        rel_count[r_idx] += 1

    new_rel_emb  = rel_emb_np.copy()
    recomputed   = 0
    for i in range(len(rel_emb_np)):
        if rel_count[i] > 0:
            new_rel_emb[i] = (rel_sum[i] / rel_count[i]).astype(np.float32)
            recomputed += 1

    label = f"[{kg_name}] " if kg_name else ""
    print(f"[lp] {label}Recomputed {recomputed}/{len(rel_emb_np)} "
          f"relation embeddings from aligned entities")
    return new_rel_emb


# ─────────────────────────────────────────────
# BUILD MERGED EMBEDDINGS AND IDX MAPPINGS
# ─────────────────────────────────────────────

def build_merged_embeddings(
    aligned_kg1_csv : str,
    aligned_kg2_csv : str,
    folder_kg1      : str,
    folder_kg2      : str,
    output_dir      : str,
    triples1        : Optional[List] = None,
    triples2        : Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """
    Build merged entity and relation embeddings.

    Entity order:
        indices 0 to n_kg1-1           → KG1 entities
        indices n_kg1 to n_kg1+n_kg2-1 → KG2 entities

    This order is preserved in entity_to_idx.
    """
    print(f"\n[lp] Building merged embeddings...")

    kg1_df = pd.read_csv(aligned_kg1_csv, index_col=0)
    kg2_df = pd.read_csv(aligned_kg2_csv, index_col=0)

    print(f"[lp] KG1 entities: {len(kg1_df)}")
    print(f"[lp] KG2 entities: {len(kg2_df)}")

    merged_ent_emb = np.vstack([
        kg1_df.values.astype(np.float32),
        kg2_df.values.astype(np.float32),
    ])

    all_uris      = list(kg1_df.index) + list(kg2_df.index)
    entity_to_idx = {uri: idx for idx, uri in enumerate(all_uris)}

    print(f"[lp] Merged entities : {len(entity_to_idx)}")
    print(f"[lp] Merged ent shape: {merged_ent_emb.shape}")

    print(f"\n[lp] Loading relation embeddings...")

    from ...data.loader import (
        extract_files_from_directory,
        _load_idx_mapping,
    )

    m1, e2i_1, r2i_1 = extract_files_from_directory(folder_kg1)
    m2, e2i_2, r2i_2 = extract_files_from_directory(folder_kg2)

    w1 = torch.load(m1, map_location="cpu", weights_only=True)
    w2 = torch.load(m2, map_location="cpu", weights_only=True)

    def get_rel_key(weights):
        for k in weights.keys():
            if "relation" in k.lower() and "weight" in k.lower():
                return k
        return None

    rel_emb1_np = w1[get_rel_key(w1)].cpu().numpy()
    rel_emb2_np = w2[get_rel_key(w2)].cpu().numpy()

    r_idx2uri_1 = _load_idx_mapping(r2i_1, kind="relation")
    r_idx2uri_2 = _load_idx_mapping(r2i_2, kind="relation")

    print(f"[lp] KG1 relations   : {len(rel_emb1_np)}")
    print(f"[lp] KG2 relations   : {len(rel_emb2_np)}")

    if triples1 is not None and triples2 is not None:
        print(f"\n[lp] Recomputing relation embeddings from aligned entities...")
        rel_emb1_np = recompute_relation_embeddings(
            rel_emb_np    = rel_emb1_np,
            r_idx2uri     = r_idx2uri_1,
            triples       = triples1,
            entity_to_idx = entity_to_idx,
            aligned_emb   = merged_ent_emb,
            kg_name       = "KG1",
        )
        rel_emb2_np = recompute_relation_embeddings(
            rel_emb_np    = rel_emb2_np,
            r_idx2uri     = r_idx2uri_2,
            triples       = triples2,
            entity_to_idx = entity_to_idx,
            aligned_emb   = merged_ent_emb,
            kg_name       = "KG2",
        )

    all_relations: Dict[str, np.ndarray] = {}
    for idx in range(len(rel_emb1_np)):
        uri = r_idx2uri_1.get(idx, f"__kg1_rel_{idx}__")
        all_relations[uri] = rel_emb1_np[idx]
    for idx in range(len(rel_emb2_np)):
        uri = r_idx2uri_2.get(idx, f"__kg2_rel_{idx}__")
        if uri in all_relations:
            all_relations[uri] = (
                all_relations[uri] + rel_emb2_np[idx]
            ) / 2.0
        else:
            all_relations[uri] = rel_emb2_np[idx]
    rel_uris_ordered = list(all_relations.keys())
    relation_to_idx  = {
        uri: idx for idx, uri in enumerate(rel_uris_ordered)
    }
    merged_rel_emb = np.array(
        [all_relations[uri] for uri in rel_uris_ordered],
        dtype=np.float32,
    )

    print(f"[lp] Merged relations: {len(relation_to_idx)}")
    print(f"[lp] Merged rel shape: {merged_rel_emb.shape}")

    fine_tune_folder = os.path.join(output_dir, "fine_tune")

    _save_idx_files(
        fine_tune_folder, entity_to_idx, relation_to_idx
    )

    print(f"[lp] Saved idx mappings to {fine_tune_folder}")

    return (
        merged_ent_emb,
        merged_rel_emb,
        entity_to_idx,
        relation_to_idx,
    )


# ─────────────────────────────────────────────
# SAVE IDX FILES HELPER
# ─────────────────────────────────────────────

def _save_idx_files(
    folder          : str,
    entity_to_idx   : Dict,
    relation_to_idx : Dict,
):
    """
    Save entity and relation idx files in both
    CSV and pickle formats. KGE needs the pickle files.
    """
    e2i_df = pd.DataFrame({
        "entity": list(entity_to_idx.keys()),
        "index" : list(entity_to_idx.values()),
    })
    e2i_df.to_csv(
        os.path.join(folder, "entity_to_idx.csv"),
        index=False
    )
    r2i_df = pd.DataFrame({
        "relation": list(relation_to_idx.keys()),
        "index"   : list(relation_to_idx.values()),
    })
    r2i_df.to_csv(
        os.path.join(folder, "relation_to_idx.csv"),
        index=False
    )

    with open(
        os.path.join(folder, "entity_to_idx.p"), "wb"
    ) as f:
        pickle.dump(entity_to_idx, f)

    with open(
        os.path.join(folder, "relation_to_idx.p"), "wb"
    ) as f:
        pickle.dump(relation_to_idx, f)


# ─────────────────────────────────────────────
# FINE TUNE KVSALL
# ─────────────────────────────────────────────

def fine_tune_kvsall(
    model           : object,
    aligned_ent_emb : np.ndarray,
    aligned_rel_emb : np.ndarray,
    train_triples   : List[Tuple],
    val_triples     : List[Tuple],
    entity_to_idx   : Dict,
    relation_to_idx : Dict,
    output_dir      : str,
    device          : torch.device,
    batch_size      : int   = 256,
    epochs          : int   = 20,
    lr              : float = 0.001,
):
    """
    Fine-tune the merged DICE model on training triples.
    Rotating 20% sample strategy from AstrA.
    Saves both CSV and pickle idx files every epoch.
    """
    from dicee import KGE, get_er_vocab, get_re_vocab
    from ...eval.eval import evaluate_link_prediction_performance

    fine_tune_folder = os.path.join(output_dir, "fine_tune")
    os.makedirs(fine_tune_folder, exist_ok=True)

    model = model.to(device)
    
    with torch.no_grad():
        model.entity_embeddings.weight.copy_(
            torch.tensor(
                aligned_ent_emb, dtype=torch.float32
            ).contiguous()
        )
        model.relation_embeddings.weight.copy_(
            torch.tensor(
                aligned_rel_emb, dtype=torch.float32
            ).contiguous()
        )

    model.entity_embeddings.weight.requires_grad  = True
    model.relation_embeddings.weight.requires_grad = True
    model.train()

    optimizer = torch.optim.Adam(
        [
            model.entity_embeddings.weight,
            model.relation_embeddings.weight,
        ],
        lr           = lr,
        weight_decay = 0.0,
    )

    def encode_triples(triples, e2i, r2i):
        encoded = []
        skipped = 0
        for h, r, t in triples:
            h_id = e2i.get(h)
            r_id = r2i.get(r)
            t_id = e2i.get(t)
            if h_id is None or r_id is None or t_id is None:
                skipped += 1
                continue
            encoded.append([h_id, r_id, t_id])
        if skipped > 0:
            print(f"[lp] Skipped {skipped} triples "
                  f"(unknown URIs)")
        return encoded

    sample_size = max(1, int(len(train_triples) * 0.2))

    print(f"\n[lp] Total train triples  : {len(train_triples)}")
    print(f"[lp] Sample per epoch     : {sample_size} (20%)")

    best_val_mrr = 0.0
    best_state   = None
    best_epoch   = 0

    val_er_vocab   = get_er_vocab(val_triples)
    val_re_vocab   = get_re_vocab(val_triples)
    train_er_vocab = get_er_vocab(train_triples)
    train_re_vocab = get_re_vocab(train_triples)

    for epoch in range(epochs):
        
        current_batch = random.sample(train_triples, sample_size)

        encoded = encode_triples(
            current_batch, entity_to_idx, relation_to_idx
        )
        random.shuffle(encoded)

        model.train()
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, len(encoded), batch_size):
            batch = encoded[i:i + batch_size]
            if not batch:
                continue

            batch_t = torch.tensor(
                batch, dtype=torch.long, device=device
            ).contiguous()

            heads      = batch_t[:, 0]
            relations  = batch_t[:, 1]
            true_tails = batch_t[:, 2]

            hr_batch = torch.stack(
                [heads, relations], dim=1
            ).contiguous()

            try:
                scores = model.forward_k_vs_all(hr_batch)
            except Exception:
                scores = model.forward_k_vs_all(batch_t)

            B, N = scores.shape
            rows = torch.arange(B, device=device)
            k    = min(256, N - 1)

            with torch.no_grad():
                s = scores.detach().clone()
                s[rows, true_tails] = float("-inf")
                hard_negs = torch.topk(s, k, dim=1).indices

            pos_scores = scores[rows, true_tails].unsqueeze(1)
            neg_scores = scores.gather(1, hard_negs)
            loss = F.relu(1.0 - pos_scores + neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)
        print(
            f"[lp] Epoch {epoch+1:02d}/{epochs} "
            f"loss={avg_loss:.4f} "
            f"(total={total_loss:.4f})"
        )

        torch.save(
            model.state_dict(),
            os.path.join(fine_tune_folder, "model.pt")
        )

        # Save both CSV and pickle — KGE needs pickle
        _save_idx_files(
            fine_tune_folder, entity_to_idx, relation_to_idx
        )

        # ── Validate ──────────────────────────────
        val_sample   = random.sample(val_triples,   min(500, len(val_triples)))
        train_sample = random.sample(train_triples, min(500, len(train_triples)))
        finetuned_model = KGE(path=fine_tune_folder)
        val_metrics = evaluate_link_prediction_performance(
            model    = finetuned_model,
            triples  = val_sample,
            er_vocab = val_er_vocab,
            re_vocab = val_re_vocab,
            quiet    = True,
        )
        train_metrics = evaluate_link_prediction_performance(
            model    = finetuned_model,
            triples  = train_sample,
            er_vocab = train_er_vocab,
            re_vocab = train_re_vocab,
            quiet    = True,
        )

        print(
            f"[lp] Train MRR={train_metrics['MRR']:.4f}  "
            f"Val H@1={val_metrics['H@1']:.4f} "
            f"H@10={val_metrics['H@10']:.4f} "
            f"MRR={val_metrics['MRR']:.4f}"
        )

        if val_metrics["MRR"] > best_val_mrr:
            best_val_mrr = val_metrics["MRR"]
            best_state   = model.state_dict()
            best_epoch   = epoch + 1
            print(f"  ✓ New best (MRR={best_val_mrr:.4f})")

    # ── Save best model ───────────────────────────
    if best_state:
        torch.save(
            best_state,
            os.path.join(fine_tune_folder, "model.pt")
        )
        # Save idx files one more time after best model
        _save_idx_files(
            fine_tune_folder, entity_to_idx, relation_to_idx
        )
        print(f"\n[lp] Best model saved (epoch {best_epoch} "
              f"val MRR={best_val_mrr:.4f})")

    return best_val_mrr


# ─────────────────────────────────────────────
# MAIN LINK PREDICTION PIPELINE
# ─────────────────────────────────────────────

def run_link_prediction_pipeline(
    aligned_kg1_csv        : str,
    aligned_kg2_csv        : str,
    folder_kg1             : str,
    folder_kg2             : str,
    train_triples          : List[Tuple],
    val_triples            : List[Tuple],
    test_triples_path      : str,
    output_dir             : str,
    device_str             : str        = "auto",
    fine_tune_epochs       : int        = 20,
    fine_tune_lr           : float      = 0.01,
    rel_triples_1_path     : Optional[str] = None,
    rel_triples_2_path     : Optional[str] = None,
) -> Dict:
    """
    Full link prediction pipeline for SAGE.

    Steps:
        1. Clean fine_tune folder
        2. Build merged entity + relation embeddings
        3. Initialize DICE model with merged embeddings
        4. Fine-tune on training triples
        5. Evaluate on test triples
    """
    from dicee import KGE, get_er_vocab, get_re_vocab
    from ...eval.eval import evaluate_link_prediction_performance

    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_str)

    print(f"\n{'='*55}")
    print(f"  LINK PREDICTION PIPELINE")
    print(f"{'='*55}")
    print(f"  Device         : {device}")
    print(f"  Fine-tune epochs: {fine_tune_epochs}")
    print(f"  Fine-tune lr   : {fine_tune_lr}")

    # ── Step 1: Clean fine_tune folder ────────────
    fine_tune_folder = os.path.join(output_dir, "fine_tune")
    if os.path.exists(fine_tune_folder):
        print(f"\n[lp] Cleaning old fine_tune folder...")
        shutil.rmtree(fine_tune_folder)
    os.makedirs(fine_tune_folder, exist_ok=True)
    print(f"[lp] Fine_tune folder ready: {fine_tune_folder}")

    # ── Step 2: Build merged embeddings ───────────
    raw_triples1 = None
    raw_triples2 = None

    if rel_triples_1_path and rel_triples_2_path:
        print(f"\n[lp] Loading raw triples for relation recomputation...")
        raw_triples1 = _load_raw_triples(rel_triples_1_path)
        raw_triples2 = _load_raw_triples(rel_triples_2_path)
        print(f"[lp] KG1 raw triples: {len(raw_triples1)}")
        print(f"[lp] KG2 raw triples: {len(raw_triples2)}")

    merged_ent_emb, merged_rel_emb, entity_to_idx, relation_to_idx = \
        build_merged_embeddings(
            aligned_kg1_csv = aligned_kg1_csv,
            aligned_kg2_csv = aligned_kg2_csv,
            folder_kg1      = folder_kg1,
            folder_kg2      = folder_kg2,
            output_dir      = output_dir,
            triples1        = raw_triples1,
            triples2        = raw_triples2,
        )

    # ── Step 3: Initialize DICE model ─────────────
    model = initialize_model_with_aligned_embeddings(
        entity_embeddings   = merged_ent_emb,
        relation_embeddings = merged_rel_emb,
        output_dir          = output_dir,
        directory_kg1       = folder_kg1,
    )

    # ── Step 4: Fine-tune ──────────────────────────
    fine_tune_kvsall(
        model           = model,
        aligned_ent_emb = merged_ent_emb,
        aligned_rel_emb = merged_rel_emb,
        train_triples   = train_triples,
        val_triples     = val_triples,
        entity_to_idx   = entity_to_idx,
        relation_to_idx = relation_to_idx,
        output_dir      = output_dir,
        device          = device,
        batch_size      = 256,
        epochs          = fine_tune_epochs,
        lr              = fine_tune_lr,
    )

    # ── Step 5: Evaluate on test triples ──────────
    print(f"\n[lp] Loading test triples: {test_triples_path}")

    test_df = pd.read_csv(
        test_triples_path,
        sep    = "\t",
        header = None,
        names  = ["subject", "relation", "object"],
        dtype  = str,
        engine = "python",
    ).dropna()
    test_triples = test_df.values.tolist()
    print(f"[lp] Test triples: {len(test_triples)}")

    er_vocab = get_er_vocab(test_triples)
    re_vocab = get_re_vocab(test_triples)

    final_model = KGE(path=fine_tune_folder)

    metrics = evaluate_link_prediction_performance(
        model    = final_model,
        triples  = test_triples,
        er_vocab = er_vocab,
        re_vocab = re_vocab,
    )

    print(f"\n{'='*55}")
    print(f"  LINK PREDICTION RESULTS")
    print(f"{'='*55}")
    print(f"  H@1  : {metrics['H@1']:.4f}")
    print(f"  H@3  : {metrics['H@3']:.4f}")
    print(f"  H@10 : {metrics['H@10']:.4f}")
    print(f"  MRR  : {metrics['MRR']:.4f}")
    print(f"{'='*55}")

    # Save results
    with open(
        os.path.join(output_dir, "lp_results.json"), "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics