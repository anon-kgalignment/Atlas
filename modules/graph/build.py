from collections import defaultdict, Counter
import logging
import torch
import torch.nn.functional as F
import numpy as np
from modules.data_loader import clean_uri
import pandas as pd



def build_graph_indexes_GCN(triples, entity_set=None):
    
    ents = set()
    for h, r, t in triples:
        ents.add(h); ents.add(t)

    ent2id = {e: i for i, e in enumerate(sorted(ents))}
    
    rels = sorted(list({r for _, r, _ in triples}))
    rel2id = {r: i for i, r in enumerate(rels)}

    adj_lists = {rel2id[r]: [] for r in rels}

    for h, r, t in triples:
        if entity_set is not None:
            if h not in entity_set or t not in entity_set:
                continue
        h_id = ent2id[h]
        t_id = ent2id[t]
        r_id = rel2id[r]
        adj_lists[r_id].append((h_id, t_id))

    class Gobj:
        pass

    G = Gobj()
    G.entity2id = ent2id
    G.relation2id = rel2id
    G.relation_set = set(rel2id.values())
    G.adj_lists = adj_lists
    
    return G


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



