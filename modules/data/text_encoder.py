
import re
import numpy as np
import torch
import os
from collections import defaultdict, Counter
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.data.graph import KGGraph


def build_dbpedia_names(attr_triples_path: str) -> Dict[str, str]:
    """
    Build text profiles for DBpedia entities.

    Profile format: "Name YEAR CONTEXT DESCRIPTION"

    Sources:
      name   : foaf:name (primary), birthName (fallback)
      year   : birthDate, releaseDate, birthYear (first 4 digits)
      context: dbo:office or dbo:background (role/occupation signal)
      desc   : dc:description — always appended when available
    """
    FOAF_NAME    = "http://xmlns.com/foaf/0.1/name"
    BIRTH_NAME   = "http://dbpedia.org/ontology/birthName"
    BIRTH_DATE   = "http://dbpedia.org/ontology/birthDate"
    RELEASE_DATE = "http://dbpedia.org/ontology/releaseDate"
    BIRTH_YEAR   = "http://dbpedia.org/ontology/birthYear"
    DC_DESC      = "http://purl.org/dc/elements/1.1/description"
    DBO_OFFICE   = "http://dbpedia.org/ontology/office"
    DBO_BG       = "http://dbpedia.org/ontology/background"

    names:   Dict[str, str] = {}
    years:   Dict[str, str] = {}
    descs:   Dict[str, str] = {}
    context: Dict[str, str] = {}

    with open(attr_triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            subj, pred, obj = parts
            obj = obj.strip('"').split("^^")[0].strip('"')

            if pred == FOAF_NAME and subj not in names:
                names[subj] = obj
            elif pred == BIRTH_NAME and subj not in names:
                names[subj] = obj
            elif pred in (BIRTH_DATE, RELEASE_DATE, BIRTH_YEAR) and subj not in years:
                year = obj[:4] if len(obj) >= 4 and obj[:4].isdigit() else None
                if year:
                    years[subj] = year
            elif pred == DC_DESC and subj not in descs and len(obj) > 5:
                descs[subj] = obj
            elif pred in (DBO_OFFICE, DBO_BG) and subj not in context:
                context[subj] = obj

    profiles: Dict[str, str] = {}
    for uri in set(names) | set(years) | set(descs) | set(context):
        parts = []
        name = names.get(uri, "")
        year = years.get(uri, "")
        ctx  = context.get(uri, "")
        desc = descs.get(uri, "")
        if name: parts.append(name)
        if year: parts.append(year)
        if ctx:  parts.append(ctx)
        if desc: parts.append(desc)
        if parts:
            profiles[uri] = " ".join(parts)

    n_full = sum(1 for u in profiles if u in names and u in years)
    n_desc = sum(1 for u in profiles if u in descs)
    n_ctx  = sum(1 for u in profiles if u in context)
    print(f"[dbpedia_profiles] {len(profiles)} profiles — "
          f"{n_full} name+year, {n_desc} with description, {n_ctx} with context")
    return profiles


def build_wikidata_names(attr_triples_path: str) -> Dict[str, str]:
    """
    Build text profiles for Wikidata entities.

    Profile format: "Name YEAR" or "Name" or "YEAR"
    Fallback for name-less entities: schema:description (never added on top of a name)

    Sources:
      name : P373 Wikipedia title (1), altLabel (2), P1476 work title (3)
      year : P569 birthDate or P577 publication date (first 4 digits)
      desc : schema:description — ONLY used when entity has no name/year at all

    Example (named):   "Vincent Price 1911"
    Example (unnamed): "American television series"
    """
    P373        = "http://www.wikidata.org/entity/P373"
    PREF_LABEL  = "http://www.w3.org/2004/02/skos/core#prefLabel"
    ALT_LABEL   = "http://www.w3.org/2004/02/skos/core#altLabel"
    P1476       = "http://www.wikidata.org/entity/P1476"
    RDFS_LABEL  = "http://www.w3.org/2000/01/rdf-schema#label"
    SCHEMA_NAME = "http://schema.org/name"
    SCHEMA_DESC = "http://schema.org/description"
    P569 = "http://www.wikidata.org/entity/P569"
    P577 = "http://www.wikidata.org/entity/P577"
    P571 = "http://www.wikidata.org/entity/P571"
    P570 = "http://www.wikidata.org/entity/P570"

    name_priority = {P373: 1, PREF_LABEL: 2, ALT_LABEL: 3, P1476: 4,
                     RDFS_LABEL: 5, SCHEMA_NAME: 6}
    year_preds    = {P569, P577, P571, P570}

    names:     Dict[str, str] = {}
    name_prio: Dict[str, int] = {}
    years:     Dict[str, str] = {}
    descs:     Dict[str, str] = {}

    with open(attr_triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            subj, pred, obj = parts
            obj = obj.strip('"').split("^^")[0].strip('"')

            if pred in name_priority:
                p = name_priority[pred]
                if subj not in name_prio or p < name_prio[subj]:
                    names[subj] = obj
                    name_prio[subj] = p
            elif pred in year_preds and subj not in years:
                year = obj[:4] if len(obj) >= 4 and obj[:4].isdigit() else None
                if year:
                    years[subj] = year
            elif pred == SCHEMA_DESC and subj not in descs and len(obj) > 5:
                descs[subj] = obj

    profiles: Dict[str, str] = {}
    for uri in set(names) | set(years) | set(descs):
        parts = []
        name = names.get(uri, "")
        year = years.get(uri, "")
        desc = descs.get(uri, "")
        if name: parts.append(name)
        if year: parts.append(year)
        if desc: parts.append(desc)
        if parts:
            profiles[uri] = " ".join(parts)

    n_full = sum(1 for u in profiles if u in names and u in years)
    n_desc = sum(1 for u in profiles if u in descs)
    print(f"[wikidata_profiles] {len(profiles)} profiles — "
          f"{n_full} with name+year, "
          f"{n_desc} with description")
    return profiles


def build_yago_names(attr_triples_path: str) -> Dict[str, str]:
    """
    Build text profiles for YAGO entities.

    Profile format: "Name YEAR"

    Sources:
      name : skos:prefLabel (primary), hasGivenName + hasFamilyName (fallback)
      year : wasBornOnDate, wasCreatedOnDate (first 4 digits)
    """
    PREF_LABEL   = "skos:prefLabel"
    GIVEN_NAME   = "hasGivenName"
    FAMILY_NAME  = "hasFamilyName"
    BORN_DATE    = "wasBornOnDate"
    CREATED_DATE = "wasCreatedOnDate"

    names:  Dict[str, str] = {}
    given:  Dict[str, str] = {}
    family: Dict[str, str] = {}
    years:  Dict[str, str] = {}

    def clean(obj: str) -> str:
        obj = obj.strip()
        if obj.startswith('"'):
            end = obj.rfind('"')
            if end > 0:
                obj = obj[1:end]
        return obj.split("^^")[0].strip()

    with open(attr_triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            subj, pred, obj = parts
            obj = clean(obj)
            if not obj:
                continue

            if pred == PREF_LABEL and subj not in names:
                names[subj] = obj
            elif pred == GIVEN_NAME and subj not in given:
                given[subj] = obj
            elif pred == FAMILY_NAME and subj not in family:
                family[subj] = obj
            elif pred in (BORN_DATE, CREATED_DATE) and subj not in years:
                year = obj[:4] if len(obj) >= 4 and obj[:4].isdigit() else None
                if year:
                    years[subj] = year

    profiles: Dict[str, str] = {}
    for uri in set(names) | set(given) | set(family) | set(years):
        name = names.get(uri, "")
        if not name:
            g = given.get(uri, "")
            f = family.get(uri, "")
            name = (g + " " + f).strip()
        year = years.get(uri, "")
        parts = []
        if name: parts.append(name)
        if year: parts.append(year)
        if parts:
            profiles[uri] = " ".join(parts)

    n_full = sum(1 for u in profiles if u in years)
    print(f"[yago_profiles] {len(profiles)} profiles — {n_full} with name+year")
    return profiles


_WIKIDATA_P_LABELS: Dict[str, str] = {
    "P161": "cast member",
    "P31" : "instance of",
    "P57" : "director",
    "P86" : "composer",
    "P54" : "member of sports team",
    "P162": "producer",
    "P175": "performer",
    "P106": "occupation",
    "P58" : "screenwriter",
    "P344": "director of photography",
    "P264": "record label",
    "P19" : "place of birth",
    "P495": "country of origin",
    "P27" : "country of citizenship",
    "P136": "genre",
    "P155": "follows",
    "P156": "followed by",
    "P17" : "country",
    "P69" : "educated at",
    "P20" : "place of death",
    "P364": "original language",
    "P750": "distributor",
    "P272": "production company",
    "P1040":"film editor",
    "P407": "language",
    "P641": "sport",
    "P166": "award received",
    "P410": "military rank",
    "P412": "voice type",
    "P190": "sister city",
}


def _decode_camel(name: str) -> str:
    return re.sub(r'([A-Z])', r' \1', name).lower().strip()


def build_relation_profile_dbpedia(rel_triples_path: str) -> Dict[str, str]:
    """
    For each DBpedia entity collect the readable names of its relation types.
    e.g. entity with dbo:birthPlace, dbo:director edges → "birth place director"
    """
    entity_rels: Dict[str, list] = defaultdict(list)
    with open(rel_triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            local = r.split("/")[-1].split("#")[-1]
            readable = _decode_camel(local)
            entity_rels[h].append(readable)
            entity_rels[t].append(readable)

    profiles: Dict[str, str] = {}
    for uri, rels in entity_rels.items():
        top = [r for r, _ in Counter(rels).most_common(8)]
        profiles[uri] = " ".join(top)

    n = sum(1 for v in profiles.values() if v)
    print(f"[dbpedia_rel_profile] {n} entities with relation profiles")
    return profiles


def build_relation_profile_wikidata(rel_triples_path: str) -> Dict[str, str]:
    entity_rels: Dict[str, list] = defaultdict(list)
    with open(rel_triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            p_num = r.split("/")[-1]
            label = _WIKIDATA_P_LABELS.get(p_num, "")
            if label:
                entity_rels[h].append(label)
                entity_rels[t].append(label)

    profiles: Dict[str, str] = {}
    for uri, rels in entity_rels.items():
        top = [r for r, _ in Counter(rels).most_common(8)]
        profiles[uri] = " ".join(top)

    n = sum(1 for v in profiles.values() if v)
    print(f"[wikidata_rel_profile] {n} entities with relation profiles")
    return profiles


def enrich_names_with_neighbors(
    names          : Dict[str, str],
    rel_triples_path: str,
    max_neighbors  : int = 3,
) -> Dict[str, str]:
    """
    For entities with empty profiles, walk 1-hop neighbours in the
    relation graph and build a profile from their names.
    Entities that already have a profile are left unchanged.
    """
    from collections import defaultdict

    # Build undirected adjacency from relation triples
    adj: Dict[str, list] = defaultdict(list)
    with open(rel_triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            h, _, t = parts[0], parts[1], parts[2]
            adj[h].append(t)
            adj[t].append(h)

    enriched = dict(names)
    n_enriched = 0

    for uri in adj:
        if enriched.get(uri, ""):   
            continue

        nbr_names = []
        for nbr in adj[uri]:
            name = names.get(nbr, "") or uri_to_text(nbr)
            name = name[:50]          # cap length
            if name and name not in nbr_names:
                nbr_names.append(name)
            if len(nbr_names) >= max_neighbors:
                break

        if nbr_names:
            enriched[uri] = " ".join(nbr_names)
            n_enriched += 1

    print(f"[neighbor_prop] {n_enriched} name-less entities enriched "
          f"from {rel_triples_path.split('/')[-1]}")
    return enriched


def uri_to_text(uri: str, names_dict: Optional[Dict] = None) -> str:
    """
    Convert entity URI to text for LaBSE encoding.

    Priority:
    1. Use names_dict if provided and has entry
    2. Extract from URI path
    3. Return empty string for blank nodes

    """
    if names_dict and uri in names_dict:
        name = names_dict[uri]
        if name:
            return name

    if "genid" in uri or "/.well-known/" in uri:
        return ""

    name = uri.split("/")[-1]

    if name.startswith("Q") and name[1:].isdigit():
        return ""


    replacements = {
        "%20": " ", "%28": "(", "%29": ")",
        "%2C": ",", "%27": "'", "%26": "&",
        "%2F": "/", "%3A": ":", "%C3%BC": "ü",
        "%C3%B6": "ö", "%C3%A4": "ä", "%C3%9F": "ß",
        "%C3%A9": "é", "%C3%A8": "è", "%C3%AA": "ê",
        "%C3%B4": "ô", "%C3%A0": "à", "%C3%B9": "ù",
        "%C3%9C": "Ü", "%C3%96": "Ö", "%C3%84": "Ä",
    }
    for enc, dec in replacements.items():
        name = name.replace(enc, dec)

    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name.strip()


class Encoder:
    """
    Sentence encoder supporting LaBSE and E5 models.

    Supported model_name values:
        "LaBSE"                                  → 768-dim, multilingual
        "intfloat/multilingual-e5-large"         → 1024-dim, multilingual (recommended)
        "intfloat/e5-large-v2"                   → 1024-dim, English-only
        "paraphrase-multilingual-mpnet-base-v2"  → 768-dim, multilingual

    """

    # E5 model name prefixes
    _E5_PREFIXES = {
        "query"  : "query: ",
        "passage": "passage: ",
    }

    def __init__(
        self,
        model_name : str = "intfloat/multilingual-e5-large",
        device     : str = "auto",
        cache_dir  : Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir  = cache_dir
        self.model      = None

        # Set output dimension based on model
        if "e5-large" in model_name:
            self.dim = 1024
        else:
            self.dim = 768

        self.is_e5 = "e5" in model_name.lower()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[encoder] Model : {model_name}")
        print(f"[encoder] Dim   : {self.dim}")
        print(f"[encoder] Device: {self.device}")
        print(f"[encoder] E5 prefix mode: {self.is_e5}")

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
            )
        print(f"[encoder] Loading model (first run may download weights)...")
        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=self.cache_dir,
            device=self.device,
        )
        print(f"[encoder] Model loaded.")

    def encode_texts(
        self,
        texts      : List[str],
        batch_size : int = 512,
        normalize  : bool = True,
        e5_role    : str  = "passage",
    ) -> np.ndarray:
        """
        Encode texts.

        Args:
            texts    : list of strings
            e5_role  : "query" or "passage" — only used for E5 models.
                       Use "passage" for entity profiles (both KGs).
        """
        self._load_model()

        empty_mask = [t == "" for t in texts]
        texts_clean = [t if t else "entity" for t in texts]

        # E5 requires prefix
        if self.is_e5:
            prefix = self._E5_PREFIXES.get(e5_role, "passage: ")
            texts_clean = [prefix + t for t in texts_clean]

        print(f"[encoder] Encoding {len(texts)} texts (batch={batch_size})...")

        embeddings = self.model.encode(
            texts_clean,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        if sum(empty_mask) > 0:
            embeddings[empty_mask] = 0.0
            print(f"[encoder] Zeroed {sum(empty_mask)} empty-profile embeddings")

        return embeddings.astype(np.float32)

    def encode_graph(
        self,
        G,
        names1    : Dict[str, str],
        names2    : Dict[str, str],
        batch_size: int = 512,
        cache_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode all entities in the merged graph.

        Returns a matrix P of shape (n_entities, 768)
        where P[i] = LaBSE embedding of entity with
        global graph ID i.

        Entities from KG1 use names1 dict.
        Entities from KG2 use names2 dict.

        Args:
            G          : MergedGraph object
            names1     : {uri: name} for KG1 entities
            names2     : {uri: name} for KG2 entities
            batch_size : LaBSE encoding batch size
            cache_path : if given, save/load from this path
                         avoids recomputing on repeated runs

        Returns:
            numpy array shape (n_entities, 768)
        """
        # ── Try loading from cache ───────────────
        if cache_path and os.path.exists(cache_path):
            print(f"[encoder] Loading cached embeddings: {cache_path}")
            cached = np.load(cache_path)
            if cached.shape[0] == G.n_entities:
                print(
                    f"[encoder] Cache loaded: shape={cached.shape}"
                )
                return cached
            else:
                print(
                    f"[encoder] Cache shape mismatch "
                    f"({cached.shape[0]} vs {G.n_entities}), "
                    f"recomputing..."
                )

        texts = []
        kg1_count = 0
        kg2_count = 0
        empty_count = 0

        for i in range(G.n_entities):
            uri = G.id2entity[i]

            if i in G.kg1_entity_ids:
                text = uri_to_text(uri, names1)
                kg1_count += 1
            else:
                text = uri_to_text(uri, names2)
                kg2_count += 1

            if not text:
                empty_count += 1

            texts.append(text)

        print(f"\n[encoder] Text extraction summary:")
        print(f"         KG1 entities : {kg1_count}")
        print(f"         KG2 entities : {kg2_count}")
        print(f"         Empty names  : {empty_count}")

        print(f"\n[encoder] Sample entity texts:")
        shown = 0
        for i in range(G.n_entities):
            if texts[i] and shown < 6:
                uri  = G.id2entity[i]
                kg   = "KG1" if i in G.kg1_entity_ids else "KG2"
                short = uri.split("/")[-1][:30]
                print(
                    f"         [{kg}] {short:30} "
                    f"→ '{texts[i]}'"
                )
                shown += 1

        P = self.encode_texts(
            texts,
            batch_size=batch_size,
            normalize=True,
        )

        print(f"\n[encoder] LaBSE matrix shape: {P.shape}")

        self._verify_alignment_signal(G, P)

        # ── Save to cache ─────────────────────────
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, P)
            print(f"[encoder] Saved to cache: {cache_path}")

        return P

    def encode_kg(
        self,
        G          : "KGGraph",
        names      : Dict[str, str],
        batch_size : int = 512,
        cache_path : Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode all entities in a single KGGraph.

        Returns P of shape (n_entities, dim) where
        P[i] = embedding of entity with local ID i.

        Args:
            G          : KGGraph (single KG)
            names      : {uri: name} dict for this KG
            batch_size : encoding batch size
            cache_path : optional cache file path

        Returns:
            numpy array shape (n_entities, dim)
        """
        if cache_path and os.path.exists(cache_path):
            print(f"[encoder] Loading cached embeddings: {cache_path}")
            cached = np.load(cache_path)
            if cached.shape[0] == G.n_entities:
                return cached
            print(f"[encoder] Cache shape mismatch, recomputing...")

        texts = [
            uri_to_text(G.id2entity[i], names)
            for i in range(G.n_entities)
        ]

        P = self.encode_texts(texts, batch_size=batch_size, normalize=True)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, P)
            print(f"[encoder] Saved to cache: {cache_path}")

        return P

    def _verify_alignment_signal(
        self,
        G,
        P         : np.ndarray,
        n_samples : int = 10,
    ):
        """
        Check cosine similarity for:
        1. Aligned pairs    → should be HIGH (>0.7)
        2. Non-aligned pairs → should be LOW (<0.3)
        """
        align_edges = G.adj_lists.get(G.align_relation_id)

        if align_edges is None or len(align_edges) == 0:
            print("[encoder] No ALIGN edges to verify")
            return

        aligned_pairs = []
        for dst, src in align_edges:
            if (dst in G.kg1_entity_ids
                    and src in G.kg2_entity_ids):
                aligned_pairs.append((dst, src))

        n = min(n_samples, len(aligned_pairs))
        sample_aligned = aligned_pairs[:n]

        aligned_sims = []
        for id1, id2 in sample_aligned:
            v1  = P[id1]
            v2  = P[id2]
            sim = float(np.dot(v1, v2))
            aligned_sims.append(sim)

        kg1_ids = list(G.kg1_entity_ids)
        kg2_ids = list(G.kg2_entity_ids)


        aligned_set = set(
            (dst, src) for dst, src in align_edges
            if dst in G.kg1_entity_ids
        )

        np.random.seed(42)
        random_kg1 = np.random.choice(kg1_ids, n * 3, replace=False)
        random_kg2 = np.random.choice(kg2_ids, n * 3, replace=False)

        non_aligned_sims = []
        for id1, id2 in zip(random_kg1, random_kg2):
            if (id1, id2) in aligned_set:
                continue
            v1  = P[id1]
            v2  = P[id2]
            sim = float(np.dot(v1, v2))
            non_aligned_sims.append(sim)
            if len(non_aligned_sims) >= n:
                break

        avg_aligned     = np.mean(aligned_sims)
        avg_non_aligned = np.mean(non_aligned_sims)
        gap             = avg_aligned - avg_non_aligned

        print(f"\n[encoder] ── Encoder Alignment Signal Check ──────")
        print(f"[encoder] Aligned pair similarity:")
        print(f"         Average : {avg_aligned:.4f}")
        print(f"         Min     : {np.min(aligned_sims):.4f}")
        print(f"         Max     : {np.max(aligned_sims):.4f}")
        print(f"\n[encoder] Non-aligned pair similarity:")
        print(f"         Average : {avg_non_aligned:.4f}")
        print(f"         Min     : {np.min(non_aligned_sims):.4f}")
        print(f"         Max     : {np.max(non_aligned_sims):.4f}")
        print(f"\n[encoder] Discriminability gap: {gap:.4f}")
        print(f"         (aligned avg - non-aligned avg)")

        if gap > 0.5:
            print(
                f"[encoder] ✓ Excellent — Encoder very discriminative"
            )
        elif gap > 0.3:
            print(
                f"[encoder] ✓ Good — Encoder clearly discriminative"
            )
        elif gap > 0.1:
            print(
                f"[encoder] ~ Moderate — Encoder somewhat discriminative"
            )
        else:
            print(
                f"[encoder] ✗ Weak — Encoder not very discriminative"
            )

        print(f"[encoder] ──────────────────────────────────────────")