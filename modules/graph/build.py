from collections import defaultdict, Counter
import logging
import torch
import torch.nn.functional as F

from modules.data_loader import clean_uri
import pandas as pd


def build_graph_indexes(triples, entity_set=None):
    TYPES = {
    "rdf:type", "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "P31", "dbo:type", "schema:category", "obo:hasCategory", "a"
    }
    LABELS = {
        "rdfs:label", "http://www.w3.org/2000/01/rdf-schema#label",
        "skos:prefLabel", "http://www.w3.org/2004/02/skos/core#prefLabel",
        "schema:name", "label", "name"
    }

    types = defaultdict(set)
    rel_hist = defaultdict(Counter)
    labels = defaultdict(set)
    adjacency = defaultdict(set)

    def ok(u): return True if entity_set is None else (u in entity_set)

    for h, r, t in triples:
        h = clean_uri(h); r = clean_uri(r); t = clean_uri(t)
        if not (ok(h) and ok(t)): 
            continue

        if r in TYPES:
            types[h].add(t)
        elif r in LABELS:
            labels[h].add(str(t).lower())
        else:
            rel_hist[h][r] += 1
            rel_hist[t][r] += 1
            adjacency[h].add(t)
            adjacency[t].add(h)

    print(f"[build_graph_indexes] Graph built with {len(adjacency)} entities, "
          f"{sum(len(v) for v in adjacency.values()) // 2} edges.")

    return {
        "types": {k: list(v) for k, v in types.items()},
        "rel_hist": {k: dict(v) for k, v in rel_hist.items()},
        "labels": {k: list(v) for k, v in labels.items()},
        "adjacency": {k: list(v) for k, v in adjacency.items()}
    }
    
    
from collections import defaultdict
import numpy as np

def build_graph_indexes_GCN(triples, entity_set=None):
    
    # STEP 1 — Collect entities
    ents = set()
    for h, r, t in triples:
        ents.add(h); ents.add(t)

    ent2id = {e: i for i, e in enumerate(sorted(ents))}
    
    # STEP 2 — Map relations
    rels = sorted(list({r for _, r, _ in triples}))
    rel2id = {r: i for i, r in enumerate(rels)}

    # STEP 3 — Build adjacency lists per relation
    adj_lists = {rel2id[r]: [] for r in rels}

    for h, r, t in triples:
        if entity_set is not None:
            if h not in entity_set or t not in entity_set:
                continue
        h_id = ent2id[h]
        t_id = ent2id[t]
        r_id = rel2id[r]
        adj_lists[r_id].append((h_id, t_id))

    print(f"[build_graph_indexes] Graph built with {len(ents)} entities, "
          f"{sum(len(v) for v in adj_lists.values())} edges.")

    # STEP 4 — Return graph object
    class Gobj:
        pass

    G = Gobj()
    G.entity2id = ent2id
    G.relation2id = rel2id
    G.relation_set = set(rel2id.values())
    G.adj_lists = adj_lists
    
    return G



def apply_neighborhood_smoothing(embeddings_df, G, alpha=0.1):
    """
    Smooth entity embeddings using adjacency from G (1-hop smoothing).
    embeddings_df: pd.DataFrame (index=entity URIs, values=embeddings)
    G: output of build_graph_indexes()
    alpha: smoothing strength (0.0–1.0)
    """
    from tqdm import tqdm
    emb = embeddings_df.copy()
    smoothed = emb.copy()
    adj = G.get("adjacency", {})

    for ent in tqdm(emb.index, desc="Neighborhood smoothing"):
        nbs = adj.get(ent, [])
        if not nbs:
            continue
        nbs = [n for n in nbs if n in emb.index]
        if not nbs:
            continue
        nb_mean = emb.loc[nbs].values.mean(axis=0)
        smoothed.loc[ent] = (1 - alpha) * emb.loc[ent].values + alpha * nb_mean
    
    return smoothed




def build_merged_graph(kg1_triples, kg2_triples):
    """
    Build a merged graph **WITHOUT alignment edges**.
    Only real triples from the two KGs.
    """
    triples = []

    # Add KG1 triples
    triples.extend(kg1_triples)

    # Add KG2 triples
    triples.extend(kg2_triples)

    print(f"[Merged graph] Total triples: {len(triples)}")

    G = build_graph_indexes_GCN(triples)

    return G


def build_merged_graph_with_alignment_edges(kg1_triples, kg2_triples, train_links):
    triples = []

    # Add KG1 + KG2 triples
    triples.extend(kg1_triples)
    triples.extend(kg2_triples)

    # Add alignment edges
    for src, tgt in train_links:
        triples.append((src, "alignment", tgt))
        triples.append((tgt, "alignment_inv", src))

    print(f"[Merged graph] Total triples with alignment edges: {len(triples)}")

    return build_graph_indexes_GCN(triples)


