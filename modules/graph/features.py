import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
import re
from sentence_transformers import SentenceTransformer

# === Load multilingual SentenceTransformer model once globally ===
_semantic_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_label_cache = {}  # cache embeddings to avoid recomputing


# === Basic helpers ===
def _cosine_topk(mat_src, mat_tgt, src_uris, tgt_uris, k=10):
    X = torch.tensor(mat_src, dtype=torch.float32)
    Y = torch.tensor(mat_tgt, dtype=torch.float32)
    X = F.normalize(X, dim=1)
    Y = F.normalize(Y, dim=1)
    sim = X @ Y.T
    vals, idxs = torch.topk(sim, k=k, dim=1)
    out = {src_uris[i]: [(tgt_uris[j.item()], vals[i, c].item()) for c, j in enumerate(idxs[i])]
           for i in range(len(src_uris))}
    return out


def _jaccard_set(a, b):
    if not a and not b:
        return 0.0
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _jaccard_counter_like(d1, d2):
    if not d1 and not d2:
        return 0.0
    keys = set(d1.keys()) | set(d2.keys())
    inter = sum(min(d1.get(k, 0), d2.get(k, 0)) for k in keys)
    union = sum(max(d1.get(k, 0), d2.get(k, 0)) for k in keys)
    return inter / union if union else 0.0


# --- Helper: tokenize URI localname ---
def _tokens_from_uri(u):
    name = u.rsplit('/', 1)[-1]
    name = name.replace('_', ' ')
    return set(re.findall(r"[A-Za-z0-9]+", name.lower()))

import urllib.parse

def _clean_type_uri(u):
    """Extract a readable type name from ontology URIs."""
    if not u:
        return ""
    u = urllib.parse.unquote(str(u))  # decode %xx
    # unify French DBpedia ontology URIs to English version
    if u.startswith("http://fr.dbpedia.org/ontology/"):
        u = u.replace("http://fr.dbpedia.org/ontology/", "http://dbpedia.org/ontology/")
    # take local name and clean it
    u = u.split('#')[-1]
    u = u.split('/')[-1]
    u = u.replace('_', ' ')
    return u.strip().lower()

# --- Helper: embed labels or types semantically ---
def _semantic_sim(text1, text2):
    """Compute cosine similarity between two short text labels using cached embeddings."""
    if not text1 or not text2:
        return 0.0
    if text1 not in _label_cache:
        _label_cache[text1] = _semantic_model.encode(text1, convert_to_tensor=True, show_progress_bar=False)
        
    if text2 not in _label_cache:
        _label_cache[text2] = _semantic_model.encode(text2, convert_to_tensor=True, show_progress_bar=False)

    return F.cosine_similarity(_label_cache[text1], _label_cache[text2], dim=0).item()


#
# --- Hybrid similarity function (cross-graph) ---
def _feature_sims_cross(s, t, Gs, Gt):
    """
    Compute hybrid (symbolic + semantic) similarity between entity s from source KG (Gs)
    and entity t from target KG (Gt).
    Returns (type_sim, rel_sim, label_sim)
    """

    # === 1) Type similarity ===
    s_types = Gs.get("types", {}).get(s, [])
    t_types = Gt.get("types", {}).get(t, [])

    # Clean and normalize URIs
    s_types_clean = [_clean_type_uri(x) for x in s_types if x]
    t_types_clean = [_clean_type_uri(x) for x in t_types if x]

    # 🔁 Fallback: if type info missing, use entity URI local name instead
    if not s_types_clean:
        s_types_clean = [_clean_type_uri(s)]
    if not t_types_clean:
        t_types_clean = [_clean_type_uri(t)]

    # Symbolic overlap (Jaccard)
    ty_jacc = _jaccard_set(s_types_clean, t_types_clean)

    # Semantic similarity using the multilingual SentenceTransformer
    ty_sem = 0.0
    if s_types_clean and t_types_clean:
        ty_sem = max(_semantic_sim(st, tt) for st in s_types_clean for tt in t_types_clean)

    ty = 0.5 * ty_jacc + 0.5 * ty_sem

    # === 2) Relation similarity ===
    rl = _jaccard_counter_like(
        Gs.get("rel_hist", {}).get(s, {}),
        Gt.get("rel_hist", {}).get(t, {})
    )

    # === 3) Label similarity ===
    s_labels = Gs.get("labels", {}).get(s, [])
    t_labels = Gt.get("labels", {}).get(t, [])

    if not s_labels:
        s_labels = [" ".join(_tokens_from_uri(s))]
    if not t_labels:
        t_labels = [" ".join(_tokens_from_uri(t))]

    lb_jacc = _jaccard_set(set(" ".join(s_labels).split()), set(" ".join(t_labels).split()))
    lb_sem = max(_semantic_sim(sl, tl) for sl in s_labels for tl in t_labels)
    lb = 0.5 * lb_jacc + 0.5 * lb_sem

    return ty, rl, lb
