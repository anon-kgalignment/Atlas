
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class KGGraph:

    def __init__(self):
        self.entity2id   = {}
        self.id2entity   = {}
        self.relation2id = {}
        self.id2relation = {}
        self.adj_lists   = defaultdict(list)
        self.n_entities  = 0
        self.n_relations = 0


def build_kg_graph(
    triples : List[Tuple[str, str, str]],
    emb     : pd.DataFrame,
    name    : str = "KG",
) -> "KGGraph":
    """
    Build a single-KG graph for the Relational GAT.

    No cross-KG edges are added. The GAT will only
    see this KG's own structure.

    Args:
        triples : list of (h_uri, r_uri, t_uri)
        emb     : entity embeddings DataFrame (index = URIs)
        name    : name for logging

    Returns:
        KGGraph object
    """
    print(f"\n[graph] Building {name} graph...")

    G = KGGraph()

    for uri in emb.index:
        if uri not in G.entity2id:
            idx = len(G.entity2id)
            G.entity2id[uri] = idx
            G.id2entity[idx] = uri
    G.n_entities = len(G.entity2id)

    all_rels = set(r for _, r, _ in triples)
    for r_uri in sorted(all_rels):
        if r_uri not in G.relation2id:
            idx = len(G.relation2id)
            G.relation2id[r_uri] = idx
            G.id2relation[idx]   = r_uri
    G.n_relations = len(G.relation2id)

    edges_added   = 0
    edges_skipped = 0
    for h_uri, r_uri, t_uri in triples:
        h_id = G.entity2id.get(h_uri)
        r_id = G.relation2id.get(r_uri)
        t_id = G.entity2id.get(t_uri)
        if h_id is None or r_id is None or t_id is None:
            edges_skipped += 1
            continue
        G.adj_lists[r_id].append((h_id, t_id))
        edges_added += 1

    for r_id in G.adj_lists:
        if len(G.adj_lists[r_id]) > 0:
            G.adj_lists[r_id] = np.array(
                G.adj_lists[r_id], dtype=np.int64
            )
        else:
            G.adj_lists[r_id] = np.empty((0, 2), dtype=np.int64)

    total_edges = sum(len(e) for e in G.adj_lists.values())
    print(f"[graph] {name}: {G.n_entities} entities, "
          f"{G.n_relations} relations, {total_edges} edges")
    if edges_skipped > 0:
        print(f"[graph]   skipped {edges_skipped} edges")

    return G


def build_embedding_matrix_single(
    G   : "KGGraph",
    emb : pd.DataFrame,
) -> np.ndarray:
    """
    Build the initial embedding matrix for a single KG.

    Args:
        G   : KGGraph
        emb : entity embeddings DataFrame (index = URIs)

    Returns:
        numpy array of shape (n_entities, dim)
    """
    dim = len(emb.columns)
    E   = np.zeros((G.n_entities, dim), dtype=np.float32)
    missing = 0
    for i in range(G.n_entities):
        uri = G.id2entity[i]
        if uri in emb.index:
            E[i] = emb.loc[uri].values
        else:
            missing += 1
    if missing > 0:
        print(f"[graph] WARNING: {missing} entities have no embedding")
    return E


def get_pair_indices_separate(
    pairs : List[Tuple[str, str]],
    G1    : "KGGraph",
    G2    : "KGGraph",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map alignment pairs to separate KG1 and KG2 local indices.

    Args:
        pairs : list of (uri_kg1, uri_kg2)
        G1    : KGGraph for KG1
        G2    : KGGraph for KG2

    Returns:
        src_idx : KG1-local entity indices
        tgt_idx : KG2-local entity indices
    """
    src_idx = []
    tgt_idx = []
    skipped = 0

    for uri1, uri2 in pairs:
        id1 = G1.entity2id.get(uri1)
        id2 = G2.entity2id.get(uri2)
        if id1 is None or id2 is None:
            skipped += 1
            continue
        src_idx.append(id1)
        tgt_idx.append(id2)

    if skipped > 0:
        print(f"[graph] Pair indexing: skipped {skipped} pairs")

    return (
        np.array(src_idx, dtype=np.int64),
        np.array(tgt_idx, dtype=np.int64),
    )



class MergedGraph:
    """
    A merged graph containing entities and relations
    from two knowledge graphs plus cross-KG bridge edges.

    Attributes:
        entity2id    : dict {uri: int}  all entities from both KGs
        id2entity    : dict {int: uri}
        relation2id  : dict {uri: int}  all relations from both KGs
                       including special ALIGN relation
        id2relation  : dict {int: uri}
        adj_lists    : dict {relation_id: list of (dst, src) pairs}
                       used by Relational GAT
        n_entities   : total number of entities
        n_relations  : total number of relations
        kg1_entity_ids : set of entity IDs belonging to KG1
        kg2_entity_ids : set of entity IDs belonging to KG2
        align_relation_id : the ID of the special ALIGN relation
    """

    def __init__(self):
        self.entity2id         = {}
        self.id2entity         = {}
        self.relation2id       = {}
        self.id2relation       = {}
        self.adj_lists         = defaultdict(list)
        self.n_entities        = 0
        self.n_relations       = 0
        self.kg1_entity_ids    = set()
        self.kg2_entity_ids    = set()
        self.align_relation_id = None


def build_merged_graph(
    triples1    : List[Tuple[str, str, str]],
    triples2    : List[Tuple[str, str, str]],
    train_pairs : List[Tuple[str, str]],
    emb1        : pd.DataFrame,
    emb2        : pd.DataFrame,
) -> MergedGraph:
    """
    Build a merged cross-KG graph for the Relational GAT.

    The merged graph contains:
        1. All KG1 entities and relations (from triples1)
        2. All KG2 entities and relations (from triples2)
        3. Special ALIGN edges between seed pair entities
           (bidirectional: KG1→KG2 and KG2→KG1)

    Important:
        We include ALL entities from emb1 and emb2
        even if they do not appear in any triple.
        This ensures every entity gets a graph node.

    Args:
        triples1    : KG1 triples (h_uri, r_uri, t_uri)
        triples2    : KG2 triples (h_uri, r_uri, t_uri)
        train_pairs : seed alignment pairs (uri_kg1, uri_kg2)
        emb1        : KG1 entity embeddings DataFrame
                      (index = entity URIs)
        emb2        : KG2 entity embeddings DataFrame
                      (index = entity URIs)

    Returns:
        MergedGraph object ready for Relational GAT
    """
    print("\n[graph] Building merged cross-KG graph...")

    G = MergedGraph()

    # ── Step 1: Register all entities ───────────
    # We register ALL entities from embeddings first
    # This ensures isolated entities (no triples) are included

    print("[graph] Registering KG1 entities...")
    for uri in emb1.index:
        if uri not in G.entity2id:
            idx = len(G.entity2id)
            G.entity2id[uri] = idx
            G.id2entity[idx] = uri
            G.kg1_entity_ids.add(idx)

    print("[graph] Registering KG2 entities...")
    for uri in emb2.index:
        if uri not in G.entity2id:
            idx = len(G.entity2id)
            G.entity2id[uri] = idx
            G.id2entity[idx] = uri
            G.kg2_entity_ids.add(idx)

    G.n_entities = len(G.entity2id)
    print(f"[graph] Total entities: {G.n_entities}")
    print(f"[graph]   KG1: {len(G.kg1_entity_ids)}")
    print(f"[graph]   KG2: {len(G.kg2_entity_ids)}")


    print("[graph] Registering relations...")

    all_relations = set()
    for _, r, _ in triples1:
        all_relations.add(r)
    for _, r, _ in triples2:
        all_relations.add(r)

    for r_uri in sorted(all_relations):
        if r_uri not in G.relation2id:
            idx = len(G.relation2id)
            G.relation2id[r_uri] = idx
            G.id2relation[idx]   = r_uri

    ALIGN_RELATION = "__ALIGN__"
    align_id = len(G.relation2id)
    G.relation2id[ALIGN_RELATION] = align_id
    G.id2relation[align_id]       = ALIGN_RELATION
    G.align_relation_id           = align_id

    G.n_relations = len(G.relation2id)
    print(f"[graph] Total relations: {G.n_relations}")
    print(f"[graph]   KG1 relations: {len(set(r for _,r,_ in triples1))}")
    print(f"[graph]   KG2 relations: {len(set(r for _,r,_ in triples2))}")
    print(f"[graph]   ALIGN relation id: {align_id}")

    print("[graph] Adding KG1 edges...")

    kg1_edges   = 0
    kg1_skipped = 0

    for h_uri, r_uri, t_uri in triples1:
        h_id = G.entity2id.get(h_uri)
        r_id = G.relation2id.get(r_uri)
        t_id = G.entity2id.get(t_uri)

        if h_id is None or r_id is None or t_id is None:
            kg1_skipped += 1
            continue

        G.adj_lists[r_id].append((h_id, t_id))
        kg1_edges += 1

    print(f"[graph]   KG1 edges added  : {kg1_edges}")
    if kg1_skipped > 0:
        print(f"[graph]   KG1 edges skipped: {kg1_skipped}")

    print("[graph] Adding KG2 edges...")

    kg2_edges   = 0
    kg2_skipped = 0

    for h_uri, r_uri, t_uri in triples2:
        h_id = G.entity2id.get(h_uri)
        r_id = G.relation2id.get(r_uri)
        t_id = G.entity2id.get(t_uri)

        if h_id is None or r_id is None or t_id is None:
            kg2_skipped += 1
            continue

        G.adj_lists[r_id].append((h_id, t_id))
        kg2_edges += 1

    print(f"[graph]   KG2 edges added  : {kg2_edges}")
    if kg2_skipped > 0:
        print(f"[graph]   KG2 edges skipped: {kg2_skipped}")

    print("[graph] Adding cross-KG bridge edges...")

    bridge_edges   = 0
    bridge_skipped = 0

    for uri_kg1, uri_kg2 in train_pairs:
        id_kg1 = G.entity2id.get(uri_kg1)
        id_kg2 = G.entity2id.get(uri_kg2)

        if id_kg1 is None or id_kg2 is None:
            bridge_skipped += 1
            continue

        G.adj_lists[align_id].append((id_kg1, id_kg2))
        G.adj_lists[align_id].append((id_kg2, id_kg1))
        bridge_edges += 2

    print(f"[graph]   Bridge edges added  : {bridge_edges}")
    print(f"[graph]   Bridge edges skipped: {bridge_skipped}")

    print("[graph] Converting adjacency lists to arrays...")

    for r_id in G.adj_lists:
        if len(G.adj_lists[r_id]) > 0:
            G.adj_lists[r_id] = np.array(
                G.adj_lists[r_id],
                dtype=np.int64
            )
        else:
            G.adj_lists[r_id] = np.empty(
                (0, 2), dtype=np.int64
            )

    total_edges = sum(
        len(edges) for edges in G.adj_lists.values()
    )

    print(f"\n[graph] ── Merged Graph Summary ─────────────")
    print(f"[graph] Total entities : {G.n_entities}")
    print(f"[graph]   KG1          : {len(G.kg1_entity_ids)}")
    print(f"[graph]   KG2          : {len(G.kg2_entity_ids)}")
    print(f"[graph] Total relations: {G.n_relations}")
    print(f"[graph]   KG1 rels     : {len(set(r for _,r,_ in triples1))}")
    print(f"[graph]   KG2 rels     : {len(set(r for _,r,_ in triples2))}")
    print(f"[graph]   ALIGN rel    : 1")
    print(f"[graph] Total edges    : {total_edges}")
    print(f"[graph]   KG1 edges    : {kg1_edges}")
    print(f"[graph]   KG2 edges    : {kg2_edges}")
    print(f"[graph]   Bridge edges : {bridge_edges}")
    print(f"[graph] ─────────────────────────────────────")

    return G


def build_embedding_matrix(
    G    : MergedGraph,
    emb1 : pd.DataFrame,
    emb2 : pd.DataFrame,
) -> np.ndarray:
    """
    Build the initial embedding matrix E for the GAT.

    E[i] = embedding of entity with global ID i
         = DICE embedding from emb1 if entity is in KG1
         = DICE embedding from emb2 if entity is in KG2

    Entities not found in either embedding get zero vector.

    Args:
        G    : MergedGraph object
        emb1 : KG1 entity embeddings DataFrame
        emb2 : KG2 entity embeddings DataFrame

    Returns:
        numpy array of shape (n_entities, dim)
    """
    dim = len(emb1.columns)
    E   = np.zeros((G.n_entities, dim), dtype=np.float32)

    missing = 0

    for i in range(G.n_entities):
        uri = G.id2entity[i]

        if uri in emb1.index:
            E[i] = emb1.loc[uri].values
        elif uri in emb2.index:
            E[i] = emb2.loc[uri].values
        else:
            missing += 1

    if missing > 0:
        print(
            f"[graph] WARNING: {missing} entities "
            f"have no embedding → zero vector used"
        )

    print(
        f"[graph] Embedding matrix built: "
        f"shape={E.shape}"
    )

    return E


def get_pair_indices(
    pairs : List[Tuple[str, str]],
    G     : MergedGraph,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert alignment pairs from URI strings to
    integer indices in the merged graph.

    Args:
        pairs : list of (uri_kg1, uri_kg2)
        G     : MergedGraph

    Returns:
        src_idx : numpy array of KG1 entity indices
        tgt_idx : numpy array of KG2 entity indices

        src_idx[i] and tgt_idx[i] are a matched pair
    """
    src_idx = []
    tgt_idx = []
    skipped = 0

    for uri1, uri2 in pairs:
        id1 = G.entity2id.get(uri1)
        id2 = G.entity2id.get(uri2)

        if id1 is None or id2 is None:
            skipped += 1
            continue

        src_idx.append(id1)
        tgt_idx.append(id2)

    if skipped > 0:
        print(
            f"[graph] Pair indexing: "
            f"skipped {skipped} pairs (not in graph)"
        )

    return (
        np.array(src_idx, dtype=np.int64),
        np.array(tgt_idx, dtype=np.int64)
    )