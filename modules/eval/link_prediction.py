# modules/eval/link_prediction.py
import torch
import sys
from tqdm import tqdm
from dicee.eval_static_funcs import evaluate_link_prediction_performance
from dicee.eval_static_funcs import evaluate_link_prediction_performance
from dicee import KGE
import json, os, pandas as pd
from dicee import get_er_vocab, get_re_vocab  
__all__ = ["evaluate_link_prediction_performance"]



@torch.no_grad()
def evaluate_link_prediction_performance(model, triples, er_vocab, re_vocab, quiet=False):
    """
    Safe evaluator: skips triples with unknown entities/relations and
    filters er_vocab/re_vocab to known entities to avoid KeyError.
    Returns {'H@1','H@3','H@10','MRR'} as usual.
    """
    # Basic checks
    model.model.eval()
    ent2idx = model.entity_to_idx
    # handle naming differences
    rel2idx = getattr(model, "relation_to_idx", None) or getattr(model, "relation_to_id", None)
    assert isinstance(ent2idx, dict) and isinstance(rel2idx, dict), "model must expose entity_to_idx / relation_to_idx dicts."

    num_entities = model.num_entities
    device = next(model.model.parameters()).device

    hits = {}
    reciprocal_ranks = []
    used = 0
    skipped = 0

    # all entities tensor on correct device
    all_entities = torch.arange(0, num_entities, dtype=torch.long, device=device)

    for i in tqdm(range(len(triples))):
        str_h, str_r, str_t = triples[i]

        # Skip if anything is unknown
        if (str_h not in ent2idx) or (str_t not in ent2idx) or (str_r not in rel2idx):
            skipped += 1
            if not quiet and skipped <= 5:  # avoid spamming
                print(f"[skip] Unknown mapping at triple {i}: "
                      f"h:{str_h in ent2idx}, r:{str_r in rel2idx}, t:{str_t in ent2idx}",
                      file=sys.stderr)
            continue

        h = ent2idx[str_h]
        r = rel2idx[str_r]
        t = ent2idx[str_t]

        h_ten = torch.tensor(h, device=device)
        r_ten = torch.tensor(r, device=device)
        t_ten = torch.tensor(t, device=device)

        # Predict tails: (h, r, ?)
        x_tails = torch.stack((
            h_ten.repeat(num_entities),
            r_ten.repeat(num_entities),
            all_entities
        ), dim=1)
        predictions_tails = model.model.forward_triples(x_tails).detach()

        # Predict heads: (?, r, t)
        x_heads = torch.stack((
            all_entities,
            r_ten.repeat(num_entities),
            t_ten.repeat(num_entities)
        ), dim=1)
        predictions_heads = model.model.forward_triples(x_heads).detach()

        # Filtered rankings for tails
        filt_tails_uris = er_vocab.get((str_h, str_r), [])
        filt_tails_idx = [ent2idx[u] for u in filt_tails_uris if (u in ent2idx) and (u != str_t)]

        target_val_tail = predictions_tails[t].item()
        if filt_tails_idx:
            predictions_tails[filt_tails_idx] = -float("inf")
        predictions_tails[t] = target_val_tail
        _, sort_idxs = torch.sort(predictions_tails, descending=True)
        filt_tail_entity_rank = (sort_idxs == t).nonzero(as_tuple=False).view(-1)[0].item() + 1

        # Filtered rankings for heads
        filt_heads_uris = re_vocab.get((str_r, str_t), [])
        filt_heads_idx = [ent2idx[u] for u in filt_heads_uris if (u in ent2idx) and (u != str_h)]

        target_val_head = predictions_heads[h].item()
        if filt_heads_idx:
            predictions_heads[filt_heads_idx] = -float("inf")
        predictions_heads[h] = target_val_head
        _, sort_idxs = torch.sort(predictions_heads, descending=True)
        filt_head_entity_rank = (sort_idxs == h).nonzero(as_tuple=False).view(-1)[0].item() + 1

        # Reciprocal ranks (both head & tail)
        rr = 1.0 / filt_head_entity_rank + 1.0 / filt_tail_entity_rank
        reciprocal_ranks.append(rr)

        for k in (1, 3, 10):
            res = (1 if filt_head_entity_rank <= k else 0) + (1 if filt_tail_entity_rank <= k else 0)
            if res > 0:
                hits.setdefault(k, []).append(res)

        used += 1

    denom = float(used * 2) if used > 0 else 1.0
    mrr = (sum(reciprocal_ranks) / denom) if used > 0 else 0.0
    hit_1 = (sum(hits.get(1, [])) / denom) if used > 0 else 0.0
    hit_3 = (sum(hits.get(3, [])) / denom) if used > 0 else 0.0
    hit_10 = (sum(hits.get(10, [])) / denom) if used > 0 else 0.0

    if not quiet:
        print(f"[eval] used={used}, skipped={skipped} (unknown entities/relations filtered).", file=sys.stderr)

    return {"H@1": hit_1, "H@3": hit_3, "H@10": hit_10, "MRR": mrr}


def run_link_prediction_evaluation(final_model_folder, finetuned_model_folder, test_triples_path, output_dir):

    full_test_triples = pd.read_csv(test_triples_path, sep="\s+", header=None,
                                    names=['subject', 'relation', 'object'], dtype=str).values.tolist()
    test_triples_1000 = full_test_triples[:1000]

    er_vocab_1000 = get_er_vocab(full_test_triples)
    re_vocab_1000 = get_re_vocab(full_test_triples)

    final_trained_model = KGE(path=final_model_folder)
    finetuned_model = KGE(path=finetuned_model_folder)

    final_metrics = evaluate_link_prediction_performance(final_trained_model,full_test_triples, er_vocab_1000, re_vocab_1000)
    finetuned_metrics = evaluate_link_prediction_performance(finetuned_model,full_test_triples, er_vocab_1000, re_vocab_1000)

    output_path = os.path.join(output_dir, "final_results.json")
    with open(output_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"Final results saved to {output_path}")
    print(f"Final Model Performance:\n{final_metrics}")
    print(f"Fine-tuned Model Performance:\n{finetuned_metrics}")
