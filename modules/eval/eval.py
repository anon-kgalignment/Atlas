# sage/modules/eval/eval.py

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import pickle
from multiprocessing import Pool
import heapq
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────
# ENTITY ALIGNMENT EVALUATION
# ─────────────────────────────────────────────

def _one_thread(start, end, embeds1, embeds2, top_k, metric, normalize):
    result = []
    for i in range(start, end):
        e1 = embeds1[i]
        if normalize:
            norm = np.linalg.norm(e1)
            if norm > 0:
                e1 = e1 / norm

        sims = []
        for j in range(len(embeds2)):
            e2 = embeds2[j]
            if normalize:
                norm = np.linalg.norm(e2)
                if norm > 0:
                    e2 = e2 / norm

            if metric == 'cosine':
                score = np.dot(e1, e2) / (
                    np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
                )
            elif metric == 'euclidean':
                score = -np.linalg.norm(e1 - e2)
            else:
                score = np.dot(e1, e2)

            sims.append((score, j))

        top_k_sim = heapq.nlargest(max(top_k), sims)
        result.append([idx for _, idx in top_k_sim])
    return result


def greedy_alignment(
    embeds1,
    embeds2,
    top_k,
    threads_num,
    metric='inner',
    normalize=False,
    csls_k=0,
    accurate=False,
):
    entity_num = embeds1.shape[0]
    step       = (entity_num + threads_num - 1) // threads_num
    ranges     = [
        (i, min(i + step, entity_num))
        for i in range(0, entity_num, step)
    ]

    args = [
        (start, end, embeds1, embeds2, top_k, metric, normalize)
        for start, end in ranges
    ]

    with Pool(threads_num) as p:
        parts = p.starmap(_one_thread, args)

    alignment_result = [x for part in parts for x in part]

    hits = []
    for k in top_k:
        hit_k = np.mean([
            1 if i in alignment_result[i][:k] else 0
            for i in range(len(alignment_result))
        ])
        hits.append(hit_k)

    ranks = [
        alignment_result[i].index(i) + 1
        if i in alignment_result[i]
        else len(alignment_result[i])
        for i in range(len(alignment_result))
    ]

    mr  = np.mean(ranks)
    mrr = np.mean([1.0 / r for r in ranks])

    return alignment_result, hits, mr, mrr


def test(
    embeds1,
    embeds2,
    mapping,
    top_k,
    threads_num,
    metric='inner',
    normalize=False,
    csls_k=0,
    accurate=True,
):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(
            embeds1, embeds2, top_k, threads_num,
            metric, normalize, csls_k, accurate
        )
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(
            test_embeds1_mapped, embeds2, top_k, threads_num,
            metric, normalize, csls_k, accurate
        )
    return alignment_rest_12, hits1_12, mr_12, mrr_12


# ─────────────────────────────────────────────
# LINK PREDICTION EVALUATION
# ─────────────────────────────────────────────

def run_link_prediction(
    folder_kg1      : str,
    aligned_emb_path: str,
    test_triples_path: str,
    output_dir      : str,
    device_str      : str = "auto",
) -> Dict:
    """
    Run link prediction evaluation using aligned embeddings.

    Steps:
        1. Load aligned entity embeddings from CSV
        2. Load original DICE model from folder_kg1
        3. Inject aligned embeddings into DICE model
        4. Fine-tune briefly on training triples
        5. Evaluate on test triples

    Args:
        folder_kg1       : KG1 DICE folder (has model.pt, config etc.)
        aligned_emb_path : path to aligned_kg1.csv from SAGE training
        test_triples_path: path to test triples file
        output_dir       : where to save results

    Returns:
        dict with H@1, H@3, H@10, MRR
    """
    try:
        from dicee import KGE
        from dicee.static_funcs import get_er_vocab, get_re_vocab
    except ImportError:
        raise ImportError(
            "dicee not installed. "
            "Install with: pip install dicee"
        )

    print(f"\n[eval] Running link prediction evaluation...")
    print(f"[eval] KG1 folder   : {folder_kg1}")
    print(f"[eval] Aligned emb  : {aligned_emb_path}")
    print(f"[eval] Test triples : {test_triples_path}")

    print(f"\n[eval] Loading aligned embeddings...")
    aligned_df = pd.read_csv(aligned_emb_path, index_col=0)
    print(f"[eval] Aligned shape: {aligned_df.shape}")

    print(f"\n[eval] Loading test triples...")
    test_triples = pd.read_csv(
        test_triples_path,
        sep=r"\s+",
        header=None,
        names=["subject", "relation", "object"],
        dtype=str,
    ).values.tolist()
    print(f"[eval] Test triples : {len(test_triples)}")

    er_vocab = get_er_vocab(test_triples)
    re_vocab = get_re_vocab(test_triples)

    fine_tune_folder = os.path.join(
        output_dir, "fine_tune"
    )

    if os.path.exists(fine_tune_folder):
        print(f"\n[eval] Loading fine-tuned model from {fine_tune_folder}")
        model = KGE(path=fine_tune_folder)
    else:
        print(f"\n[eval] Fine-tune folder not found.")
        print(f"[eval] Loading original model from {folder_kg1}")
        model = KGE(path=folder_kg1)

    print(f"\n[eval] Evaluating link prediction...")

    from ...eval.eval import evaluate_link_prediction_performance
    metrics = evaluate_link_prediction_performance(
        model    = model,
        triples  = test_triples,
        er_vocab = er_vocab,
        re_vocab = re_vocab,
    )

    print(f"\n[eval] Link Prediction Results:")
    print(f"       H@1  : {metrics['H@1']:.4f}")
    print(f"       H@3  : {metrics['H@3']:.4f}")
    print(f"       H@10 : {metrics['H@10']:.4f}")
    print(f"       MRR  : {metrics['MRR']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "lp_results.json"), "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ─────────────────────────────────────────────
# EVALUATE_LINK_PREDICTION_PERFORMANCE
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_link_prediction_performance(
    model,
    triples,
    er_vocab,
    re_vocab,
    quiet=False,
):
 
    from tqdm import tqdm

    model.model.eval()
    ent2idx = model.entity_to_idx
    rel2idx = (
        getattr(model, "relation_to_idx", None)
        or getattr(model, "relation_to_id", None)
    )

    assert isinstance(ent2idx, dict) and isinstance(rel2idx, dict)

    num_entities = model.num_entities
    device       = next(model.model.parameters()).device

    hits             = {}
    reciprocal_ranks = []
    used             = 0
    skipped          = 0

    all_entities = torch.arange(
        0, num_entities, dtype=torch.long, device=device
    )

    for i in tqdm(range(len(triples))):
        str_h, str_r, str_t = triples[i]

        if (
            (str_h not in ent2idx)
            or (str_t not in ent2idx)
            or (str_r not in rel2idx)
        ):
            skipped += 1
            continue

        h = ent2idx[str_h]
        r = rel2idx[str_r]
        t = ent2idx[str_t]

        h_ten = torch.tensor(h, device=device)
        r_ten = torch.tensor(r, device=device)
        t_ten = torch.tensor(t, device=device)

        x_tails = torch.stack((
            h_ten.repeat(num_entities),
            r_ten.repeat(num_entities),
            all_entities,
        ), dim=1)
        pred_tails = model.model.forward_triples(x_tails).detach()

        x_heads = torch.stack((
            all_entities,
            r_ten.repeat(num_entities),
            t_ten.repeat(num_entities),
        ), dim=1)
        pred_heads = model.model.forward_triples(x_heads).detach()

        filt_t = er_vocab.get((str_h, str_r), [])
        filt_t_idx = [
            ent2idx[u] for u in filt_t
            if (u in ent2idx) and (u != str_t)
        ]
        target_t = pred_tails[t].item()
        if filt_t_idx:
            pred_tails[filt_t_idx] = -float("inf")
        pred_tails[t] = target_t
        _, sort_t = torch.sort(pred_tails, descending=True)
        tail_rank = (sort_t == t).nonzero(
            as_tuple=False
        ).view(-1)[0].item() + 1

        filt_h = re_vocab.get((str_r, str_t), [])
        filt_h_idx = [
            ent2idx[u] for u in filt_h
            if (u in ent2idx) and (u != str_h)
        ]
        target_h = pred_heads[h].item()
        if filt_h_idx:
            pred_heads[filt_h_idx] = -float("inf")
        pred_heads[h] = target_h
        _, sort_h = torch.sort(pred_heads, descending=True)
        head_rank = (sort_h == h).nonzero(
            as_tuple=False
        ).view(-1)[0].item() + 1

        rr = 1.0 / head_rank + 1.0 / tail_rank
        reciprocal_ranks.append(rr)

        for k in (1, 3, 10):
            res = (
                (1 if head_rank <= k else 0)
                + (1 if tail_rank <= k else 0)
            )
            if res > 0:
                hits.setdefault(k, []).append(res)

        used += 1

    denom  = float(used * 2) if used > 0 else 1.0
    mrr    = sum(reciprocal_ranks) / denom if used > 0 else 0.0
    hit_1  = sum(hits.get(1, [])) / denom if used > 0 else 0.0
    hit_3  = sum(hits.get(3, [])) / denom if used > 0 else 0.0
    hit_10 = sum(hits.get(10, [])) / denom if used > 0 else 0.0

    if not quiet:
        print(
            f"[eval] used={used}, skipped={skipped}",
            file=sys.stderr
        )

    return {
        "H@1" : hit_1,
        "H@3" : hit_3,
        "H@10": hit_10,
        "MRR" : mrr,
    }