# sage/modules/data/loader.py

import os
import re
import json
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple


def clean_uri(uri: str) -> str:
    """
    Clean a URI string.
    Removes angle brackets, whitespace, invisible chars.

    """
    if not isinstance(uri, str):
        return ""
    uri = uri.replace("<", "").replace(">", "")
    uri = uri.strip().strip("\t")
    uri = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', uri)
    return uri


def extract_entity_name(uri: str) -> str:
    """
    Extract human readable name from URI for LaBSE.

    Examples:
        http://dbpedia.org/resource/Barack_Obama
            → Barack Obama
        http://de.dbpedia.org/resource/Sheffield_Wednesday
            → Sheffield Wednesday
        genid/N46c... → ""
    """
    if not isinstance(uri, str) or uri == "":
        return ""

    if "genid" in uri or "/.well-known/" in uri:
        return ""

    name = uri.split("/")[-1]

    url_encodings = {
        "%20": " ", "%28": "(", "%29": ")",
        "%2C": ",", "%27": "'", "%26": "&",
        "%2F": "/", "%3A": ":", "%C3%BC": "ü",
        "%C3%B6": "ö", "%C3%A4": "ä", "%C3%9F": "ß",
        "%C3%A9": "é", "%C3%A8": "è", "%C3%AA": "ê",
        "%C3%B4": "ô", "%C3%A0": "à", "%C3%B9": "ù",
    }
    for encoded, decoded in url_encodings.items():
        name = name.replace(encoded, decoded)

    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name.strip()

def extract_files_from_directory(directory: str):
    """
    Find required files in a DICE output folder.

    Looks for:
        model.pt
        entity_to_idx  (.p, .csv, no extension)
        relation_to_idx (.p, .csv, no extension)

    Returns:
        model_path, entity_to_id_path, relation_to_id_path
    """
    model_path = os.path.join(directory, "model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.pt not found in {directory}"
        )

    entity_options = [
        os.path.join(directory, "entity_to_idx.p"),
        os.path.join(directory, "entity_to_idx.pkl"),
        os.path.join(directory, "entity_to_idx.csv"),
        os.path.join(directory, "entity_to_idx"),
    ]

    entity_to_id_path = next(
        (p for p in entity_options if os.path.exists(p)),
        None
    )

    if entity_to_id_path is None:
        raise FileNotFoundError(
            f"entity_to_idx file not found in {directory}.\n"
            f"Checked: {entity_options}"
        )

    relation_options = [
        os.path.join(directory, "relation_to_idx.p"),
        os.path.join(directory, "relation_to_idx.pkl"),
        os.path.join(directory, "relation_to_idx.csv"),
        os.path.join(directory, "relation_to_idx"),
    ]

    relation_to_id_path = next(
        (p for p in relation_options if os.path.exists(p)),
        None
    )

    if relation_to_id_path is None:
        raise FileNotFoundError(
            f"relation_to_idx file not found in {directory}.\n"
            f"Checked: {relation_options}"
        )

    logging.info(f"[loader] model          : {model_path}")
    logging.info(f"[loader] entity_to_idx  : {entity_to_id_path}")
    logging.info(f"[loader] relation_to_idx: {relation_to_id_path}")

    return model_path, entity_to_id_path, relation_to_id_path


def _load_idx_mapping(path: str, kind: str = "entity") -> Dict[int, str]:
    """
    Load an index → URI mapping from any supported format.

    kind: "entity" or "relation" 
          used to find the right column if CSV

    Handles:
        .p / .pkl → pickle
        .csv / no extension → CSV file

    Returns:
        dict {int_index: uri_string}
        so you can do: uri = mapping[0], mapping[1], ...
    """
    ext = os.path.splitext(path)[1].lower()
    is_pickle = ext in (".p", ".pkl", ".pickle")
    
    if is_pickle or (ext == "" and _is_pickle_file(path)):
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            sample_key = next(iter(obj))

            if isinstance(sample_key, str):
    
                return {
                    int(v): clean_uri(str(k))
                    for k, v in obj.items()
                }
            else:
        
                return {
                    int(k): clean_uri(str(v))
                    for k, v in obj.items()
                }


        elif isinstance(obj, pd.DataFrame):
            return _extract_mapping_from_df(obj, kind)

        elif isinstance(obj, (list, tuple)):
            result = {}
            for item in obj:
                if len(item) >= 2:
    
                    a, b = item[0], item[1]
                    if isinstance(a, int):
                        result[a] = clean_uri(str(b))
                    elif isinstance(b, int):
                        result[int(b)] = clean_uri(str(a))
                    else:
                 
                        try:
                            result[int(b)] = clean_uri(str(a))
                        except (ValueError, TypeError):
                            pass
            return result

        else:
            raise ValueError(
                f"Unsupported pickle content type: {type(obj)}"
            )


    else:
        df = _read_csv_robust(path)
        return _extract_mapping_from_df(df, kind)


def _is_pickle_file(path: str) -> bool:
    """Check if a file is a pickle by reading its magic bytes."""
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
    
        return magic[0] == 0x80
    except Exception:
        return False


def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    Read a CSV file trying multiple separators.
    Returns a DataFrame.
    """

    for sep in ["\t", ",", " "]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                dtype=str,
                encoding="utf-8",
                on_bad_lines="skip",
            )

            if len(df.columns) >= 2:
                return df
        except Exception:
            continue


    return pd.read_csv(path, dtype=str, encoding="utf-8")


def _extract_mapping_from_df(
    df: pd.DataFrame,
    kind: str
) -> Dict[int, str]:
    """
    Extract {int_index: uri_string} from a DataFrame.
    """
    cols = [str(c).lower() for c in df.columns]
    original_cols = list(df.columns)

    uri_candidates_entity   = [
        "entity", "uri", "name", "label",
        "subject", "resource", "url"
    ]

    uri_candidates_relation = [
        "relation", "predicate", "uri",
        "name", "label", "url"
    ]

    uri_candidates = (
        uri_candidates_entity
        if kind == "entity"
        else uri_candidates_relation
    )

    uri_col = None
    for candidate in uri_candidates:
        if candidate in cols:
            uri_col = original_cols[cols.index(candidate)]
            break


    idx_candidates = ["index", "idx", "id", "i", "num"]

    idx_col = None
    for candidate in idx_candidates:
        if candidate in cols:
            idx_col = original_cols[cols.index(candidate)]
            break

    if uri_col is None or idx_col is None:
        col0 = original_cols[0]
        col1 = original_cols[1] if len(original_cols) > 1 else None
        
        sample0 = str(df[col0].iloc[0]).strip() if len(df) > 0 else ""
        sample1 = str(df[col1].iloc[0]).strip() if (
            col1 and len(df) > 0
        ) else ""

        def looks_like_uri(s):
            return (
                s.startswith("http")
                or s.startswith("<")
                or "/" in s
                or "_" in s
            )

        def looks_like_int(s):
            try:
                int(float(s))
                return True
            except (ValueError, TypeError):
                return False

        if looks_like_uri(sample0) and looks_like_int(sample1):
            uri_col = col0
            idx_col = col1
        elif looks_like_int(sample0) and looks_like_uri(sample1):
            uri_col = col1
            idx_col = col0
        elif looks_like_uri(sample0):
            uri_col = col0
            idx_col = None 
        else:
            idx_col = col0
            uri_col = col1


    result = {}

    for row_num, (_, row) in enumerate(df.iterrows()):
        uri = clean_uri(str(row[uri_col])) if uri_col else ""

        if idx_col:
            try:
                idx = int(float(str(row[idx_col])))
            except (ValueError, TypeError):
                idx = row_num
        else:
            idx = row_num

        if uri:
            result[idx] = uri

    return result



def load_embeddings(
    model_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load entity and relation embeddings from DICE model.

    Extracts weights directly from model.pt and maps them
    to URI strings using entity_to_idx and relation_to_idx.

    Handles all file format variations for idx mappings:
        .p / .pkl → pickle (dict, DataFrame, list)
        .csv      → CSV with any column names
        no ext    → auto-detected (pickle or CSV)

    Args:
        model_path          : path to model.pt
        entity_to_id_path   : path to entity_to_idx file
        relation_to_id_path : path to relation_to_idx file

    Returns:
        entity_embeddings_df   : DataFrame (URI index, float values)
        relation_embeddings_df : DataFrame (URI index, float values)
    """
    logging.info(f"[loader] Loading model: {model_path}")

    model_weights = torch.load(
        model_path,
        map_location="cpu",
    )


    entity_key   = None
    relation_key = None

    for key in model_weights.keys():
        k = key.lower()
        if "entity" in k and "weight" in k:
            entity_key = key
        if "relation" in k and "weight" in k:
            relation_key = key

    if entity_key is None:
        raise KeyError(
            f"No entity embedding key found in model.pt. "
            f"Available keys: {list(model_weights.keys())}"
        )

    entity_embeddings   = model_weights[entity_key].cpu().detach().numpy()
    relation_embeddings = model_weights[relation_key].cpu().detach().numpy()

    logging.info(
        f"[loader] Entity embeddings shape  : {entity_embeddings.shape}"
    )
    logging.info(
        f"[loader] Relation embeddings shape: {relation_embeddings.shape}"
    )

    entity_idx_to_uri   = _load_idx_mapping(
        entity_to_id_path, kind="entity"
    )
    relation_idx_to_uri = _load_idx_mapping(
        relation_to_id_path, kind="relation"
    )

    n_entities   = len(entity_embeddings)
    n_relations  = len(relation_embeddings)

    sorted_entities = []
    for i in range(n_entities):
        uri = entity_idx_to_uri.get(i, f"__unknown_entity_{i}__")
        sorted_entities.append(clean_uri(uri))

    sorted_relations = []
    for i in range(n_relations):
        uri = relation_idx_to_uri.get(i, f"__unknown_relation_{i}__")
        sorted_relations.append(clean_uri(uri))

    n_unknown_ent = sum(
        1 for u in sorted_entities
        if u.startswith("__unknown_entity_")
    )
    n_unknown_rel = sum(
        1 for u in sorted_relations
        if u.startswith("__unknown_relation_")
    )

    if n_unknown_ent > 0:
        logging.warning(
            f"[loader] {n_unknown_ent} entity indices "
            f"have no URI mapping"
        )
    if n_unknown_rel > 0:
        logging.warning(
            f"[loader] {n_unknown_rel} relation indices "
            f"have no URI mapping"
        )

    entity_embeddings_df = pd.DataFrame(
        entity_embeddings,
        index=sorted_entities,
        dtype=np.float32
    )

    relation_embeddings_df = pd.DataFrame(
        relation_embeddings,
        index=sorted_relations,
        dtype=np.float32
    )

    entity_embeddings_df = entity_embeddings_df[
        ~entity_embeddings_df.index.str.startswith(
            "__unknown_entity_"
        )
    ]
    relation_embeddings_df = relation_embeddings_df[
        ~relation_embeddings_df.index.str.startswith(
            "__unknown_relation_"
        )
    ]

    entity_embeddings_df = entity_embeddings_df[
        ~entity_embeddings_df.index.duplicated(keep="first")
    ]
    relation_embeddings_df = relation_embeddings_df[
        ~relation_embeddings_df.index.duplicated(keep="first")
    ]

    logging.info(
        f"[loader] Entity embeddings loaded  : "
        f"{len(entity_embeddings_df)}"
    )
    logging.info(
        f"[loader] Relation embeddings loaded: "
        f"{len(relation_embeddings_df)}"
    )

    return entity_embeddings_df, relation_embeddings_df


# ─────────────────────────────────────────────
# LOAD ALIGNMENT PAIRS
# ─────────────────────────────────────────────

def load_alignment_pairs(
    path: str
) -> List[Tuple[str, str]]:
    """
    Load alignment pairs from any supported format.

    Handles:
        .tsv → tab separated (your main format)
        .csv → comma separated
        .txt → various separators (=>, tab, space)
        .p   → pickle list of tuples or dict

    Returns:
        List of (uri_kg1, uri_kg2) tuples, both cleaned
    """
    ext = os.path.splitext(path)[1].lower()
    pairs = []

    # ── PICKLE ──────────────────────────────────
    if ext in (".p", ".pkl") or (
        ext == "" and _is_pickle_file(path)
    ):
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, (list, tuple)):
            for item in obj:
                if len(item) >= 2:
                    u1 = clean_uri(str(item[0]))
                    u2 = clean_uri(str(item[1]))
                    if u1 and u2 and u1 != u2:
                        pairs.append((u1, u2))

        elif isinstance(obj, dict):
            for k, v in obj.items():
                u1 = clean_uri(str(k))
                u2 = clean_uri(str(v))
                if u1 and u2 and u1 != u2:
                    pairs.append((u1, u2))

        elif isinstance(obj, pd.DataFrame):
            cols = list(obj.columns)
            for _, row in obj.iterrows():
                u1 = clean_uri(str(row[cols[0]]))
                u2 = clean_uri(str(row[cols[1]]))
                if u1 and u2 and u1 != u2:
                    pairs.append((u1, u2))

    # ── TEXT FORMATS ─────────────────────────────
    else:
        sep = (
            "\t" if ext == ".tsv"
            else "," if ext == ".csv"
            else None
        )
        skipped = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip("\n").rstrip("\r")

                if not line.strip():
                    continue
                if line.strip().startswith("#"):
                    continue

                stripped = line.strip()
                if stripped.upper().startswith("PAIR:"):
                    stripped = stripped[5:].strip()

                # Detect separator
                parts = None
                if sep == "\t" or (
                    sep is None and "\t" in stripped
                ):
                    parts = stripped.split("\t")
                elif sep == ",":
                    parts = stripped.split(",", 1)
                elif "=>" in stripped:
                    parts = stripped.split("=>", 1)
                elif "  " in stripped:
                    parts = stripped.split("  ", 1)

                if not parts or len(parts) < 2:
                    skipped += 1
                    if skipped <= 3:
                        logging.warning(
                            f"[loader] Cannot parse line "
                            f"{line_num}: {repr(stripped[:60])}"
                        )
                    continue

                u1 = clean_uri(parts[0])
                u2 = clean_uri(parts[1])

                if not u1 or not u2 or u1 == u2:
                    skipped += 1
                    continue

                pairs.append((u1, u2))

        if skipped > 0:
            logging.info(f"[loader] Skipped {skipped} lines")

    logging.info(f"[loader] Alignment pairs loaded: {len(pairs)}")
    return pairs


# ─────────────────────────────────────────────
# FILTER VALID PAIRS
# ─────────────────────────────────────────────

def filter_valid_pairs(
    pairs: List[Tuple[str, str]],
    emb1: pd.DataFrame,
    emb2: pd.DataFrame,
) -> List[Tuple[str, str]]:
    set1 = set(emb1.index)
    set2 = set(emb2.index)

    valid     = []
    miss1     = 0
    miss2     = 0
    miss_both = 0

    for u1, u2 in pairs:
        in1 = u1 in set1
        in2 = u2 in set2
        if in1 and in2:
            valid.append((u1, u2))
        elif not in1 and not in2:
            miss_both += 1
        elif not in1:
            miss1 += 1
        else:
            miss2 += 1

    total = len(pairs)

    print(f"\n[loader] ── Pair Filtering ──────────────────")
    print(f"[loader] Total    : {total}")
    print(f"[loader] Valid    : {len(valid)}")
    print(f"[loader] Miss KG1 : {miss1}")
    print(f"[loader] Miss KG2 : {miss2}")
    print(f"[loader] Miss both: {miss_both}")
    if total > 0:
        print(f"[loader] Coverage : {len(valid)/total*100:.1f}%")
    print(f"[loader] ─────────────────────────────────────")

    # Debug if no valid pairs
    if len(valid) == 0:
        print(f"\n[loader] !! NO VALID PAIRS FOUND !!")
        print(f"\nPair KG1 samples:")
        for u1, _ in pairs[:3]:
            print(f"  {repr(u1)}")
        print(f"\nEmbedding KG1 samples:")
        for u in list(emb1.index)[:3]:
            print(f"  {repr(u)}")
        print(f"\nPair KG2 samples:")
        for _, u2 in pairs[:3]:
            print(f"  {repr(u2)}")
        print(f"\nEmbedding KG2 samples:")
        for u in list(emb2.index)[:3]:
            print(f"  {repr(u)}")

    return valid


# ─────────────────────────────────────────────
# SPLIT PAIRS
# ─────────────────────────────────────────────

def split_pairs(
    pairs: List[Tuple[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    np.random.seed(seed)
    idx = np.random.permutation(len(pairs))
    n       = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = [pairs[i] for i in idx[:n_train]]
    val   = [pairs[i] for i in idx[n_train: n_train + n_val]]
    test  = [pairs[i] for i in idx[n_train + n_val:]]

    print(f"\n[loader] Split → "
          f"Train:{len(train)} Val:{len(val)} Test:{len(test)}")

    return train, val, test


# ─────────────────────────────────────────────
# LOAD TRIPLES FROM train_set.npy
# ─────────────────────────────────────────────

def load_triples_from_npy(
    npy_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
) -> List[Tuple[str, str, str]]:
    """
    Load triples from train_set.npy and convert
    integer indices to URI strings.

    Args:
        npy_path            : path to train_set.npy
        entity_to_id_path   : path to entity_to_idx file
        relation_to_id_path : path to relation_to_idx file

    Returns:
        List of (subject_uri, relation_uri, object_uri)
    """
    if not os.path.exists(npy_path):
        logging.warning(f"[loader] train_set.npy not found: {npy_path}")
        return []

    arr = np.load(npy_path)
    logging.info(f"[loader] train_set.npy shape: {arr.shape}")

    # Load index → URI mappings
    e_map = _load_idx_mapping(entity_to_id_path,   kind="entity")
    r_map = _load_idx_mapping(relation_to_id_path, kind="relation")

    triples = []
    skipped = 0

    for row in arr:
        if len(row) < 3:
            skipped += 1
            continue

        h_uri = e_map.get(int(row[0]), "")
        r_uri = r_map.get(int(row[1]), "")
        t_uri = e_map.get(int(row[2]), "")

        if not h_uri or not r_uri or not t_uri:
            skipped += 1
            continue

        triples.append((h_uri, r_uri, t_uri))

    logging.info(f"[loader] Triples loaded : {len(triples)}")
    if skipped > 0:
        logging.warning(f"[loader] Triples skipped: {skipped}")

    return triples


# ─────────────────────────────────────────────
# MAIN LOAD FUNCTION
# ─────────────────────────────────────────────

def load_all(
    folder_kg1      : str,
    folder_kg2      : str,
    # Option 1: direct file paths (your case)
    train_links     : Optional[str] = None,
    val_links       : Optional[str] = None,
    test_links      : Optional[str] = None,
    # Option 2: directory with pre-split files
    alignment_dir   : Optional[str] = None,
    # Option 3: single file auto-split
    alignment_path  : Optional[str] = None,
    train_ratio     : float = 0.7,
    val_ratio       : float = 0.15,
    test_ratio      : float = 0.15,
) -> Dict:
    """
    Load everything for ATLAS

    Three ways to provide alignment pairs:

    Option 1 — Direct file paths (your current case):
        train_links = ".../train_links"
        val_links   = ".../valid_links"
        test_links  = ".../test_links"

    Option 2 — Directory with named files:
        alignment_dir = ".../EN_DE_15K/"
        expects train.tsv, valid.tsv, test.tsv inside

    Option 3 — Single file, auto-split:
        alignment_path = ".../all_pairs.tsv"
    """
    print("\n" + "=" * 50)
    print("  ATLAS — DATA LOADING")
    print("=" * 50)

    # ── Load KG1 ─────────────────────────────────
    print(f"\n[loader] Loading KG1: {folder_kg1}")
    m1, e2i_1, r2i_1 = extract_files_from_directory(folder_kg1)
    emb1, rel_emb1   = load_embeddings(m1, e2i_1, r2i_1)
    npy1             = os.path.join(folder_kg1, "train_set.npy")
    triples1         = load_triples_from_npy(npy1, e2i_1, r2i_1)

    config1 = {}
    cfg_path = os.path.join(folder_kg1, "configuration.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            config1 = json.load(f)
    model_type = config1.get("model", "TransE")

    # ── Load KG2 ─────────────────────────────────
    print(f"\n[loader] Loading KG2: {folder_kg2}")
    m2, e2i_2, r2i_2 = extract_files_from_directory(folder_kg2)
    emb2, rel_emb2   = load_embeddings(m2, e2i_2, r2i_2)
    npy2             = os.path.join(folder_kg2, "train_set.npy")
    triples2         = load_triples_from_npy(npy2, e2i_2, r2i_2)

    # ── Load alignment pairs ──────────────────────

    # Option 1: Direct file paths
    if train_links is not None:
        print(f"\n[loader] Loading pre-split files directly...")
        print(f"[loader] Train: {train_links}")
        print(f"[loader] Val  : {val_links}")
        print(f"[loader] Test : {test_links}")

        train_raw   = load_alignment_pairs(train_links)
        val_raw     = load_alignment_pairs(val_links)
        test_raw    = load_alignment_pairs(test_links)

        train_pairs = filter_valid_pairs(train_raw, emb1, emb2)
        val_pairs   = filter_valid_pairs(val_raw,   emb1, emb2)
        test_pairs  = filter_valid_pairs(test_raw,  emb1, emb2)
        all_pairs   = train_pairs + val_pairs + test_pairs

    # Option 2: Directory with named files
    elif alignment_dir is not None:
        print(f"\n[loader] Loading from alignment dir: {alignment_dir}")
        train_pairs, val_pairs, test_pairs = load_alignment_from_dir(
            alignment_dir, emb1, emb2
        )
        all_pairs = train_pairs + val_pairs + test_pairs

    # Option 3: Single file auto-split
    elif alignment_path is not None:
        print(f"\n[loader] Loading single alignment file: {alignment_path}")
        raw_pairs   = load_alignment_pairs(alignment_path)
        all_pairs   = filter_valid_pairs(raw_pairs, emb1, emb2)

        if len(all_pairs) == 0:
            raise ValueError(
            
            )

        train_pairs, val_pairs, test_pairs = split_pairs(
            all_pairs, train_ratio, val_ratio, test_ratio
        )

    else:
        raise ValueError(
       
        )

    # ── Entity names for encoder ────────────────────
    names1 = {uri: extract_entity_name(uri) for uri in emb1.index}
    names2 = {uri: extract_entity_name(uri) for uri in emb2.index}

    n1 = sum(1 for n in names1.values() if n)
    n2 = sum(1 for n in names2.values() if n)

    print(f"\n[loader] encoder coverage:")
    print(f"         KG1: {n1}/{len(names1)} ({n1/len(names1)*100:.1f}%)")
    print(f"         KG2: {n2}/{len(names2)} ({n2/len(names2)*100:.1f}%)")

    print("\n" + "=" * 50)
    print("  LOADING COMPLETE")
    print("=" * 50)
    print(f"  Model type   : {model_type}")
    print(f"  Dim          : {len(emb1.columns)}")
    print(f"  KG1 entities : {len(emb1)}")
    print(f"  KG2 entities : {len(emb2)}")
    print(f"  KG1 triples  : {len(triples1)}")
    print(f"  KG2 triples  : {len(triples2)}")
    print(f"  Train pairs  : {len(train_pairs)}")
    print(f"  Val pairs    : {len(val_pairs)}")
    print(f"  Test pairs   : {len(test_pairs)}")
    print("=" * 50)

    return {
        "emb1"        : emb1,
        "emb2"        : emb2,
        "rel_emb1"    : rel_emb1,
        "rel_emb2"    : rel_emb2,
        "triples1"    : triples1,
        "triples2"    : triples2,
        "all_pairs"   : all_pairs,
        "train_pairs" : train_pairs,
        "val_pairs"   : val_pairs,
        "test_pairs"  : test_pairs,
        "names1"      : names1,
        "names2"      : names2,
        "model_type"  : model_type,
        "dim"         : len(emb1.columns),
        "n_entities_1": len(emb1),
        "n_entities_2": len(emb2),
        "folder_kg1"  : folder_kg1,
        "folder_kg2"  : folder_kg2,
        "config_kg1"  : config1,
    }
    
def load_alignment_from_dir(
    alignment_dir: str,
    emb1: pd.DataFrame,
    emb2: pd.DataFrame,
) -> Tuple[List, List, List]:
    """
    Load pre-split alignment pairs from a directory.

    Expects files named:
        train.tsv / train.txt / train.csv
        valid.tsv / val.tsv / valid.txt / val.txt
        test.tsv  / test.txt  / test.csv

    Args:
        alignment_dir : directory containing split files
        emb1          : KG1 embeddings (for filtering)
        emb2          : KG2 embeddings (for filtering)

    Returns:
        train_pairs, val_pairs, test_pairs
    """
    print(f"\n[loader] Loading pre-split alignments from: {alignment_dir}")

    def find_file(directory, names):
        """Find first existing file from list of candidate names."""
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
        return None

    # Find train file
    train_path = find_file(alignment_dir, [
        "train.tsv", "train.txt", "train.csv",
        "train_links.tsv", "train_links.txt",
    ])

    # Find val file
    val_path = find_file(alignment_dir, [
        "valid.tsv", "val.tsv", "valid.txt", "val.txt",
        "valid.csv", "val.csv",
        "valid_links.tsv", "valid_links.txt",
    ])

    # Find test file
    test_path = find_file(alignment_dir, [
        "test.tsv", "test.txt", "test.csv",
        "test_links.tsv", "test_links.txt",
    ])

    if train_path is None:
        raise FileNotFoundError(
            f"No train file found in {alignment_dir}. "
            f"Expected train.tsv / train.txt / train.csv"
        )
    if val_path is None:
        raise FileNotFoundError(
            f"No val file found in {alignment_dir}. "
            f"Expected valid.tsv / val.tsv / valid.txt"
        )
    if test_path is None:
        raise FileNotFoundError(
            f"No test file found in {alignment_dir}. "
            f"Expected test.tsv / test.txt / test.csv"
        )

    print(f"[loader] Train file : {os.path.basename(train_path)}")
    print(f"[loader] Val file   : {os.path.basename(val_path)}")
    print(f"[loader] Test file  : {os.path.basename(test_path)}")

    # Load each split
    train_raw = load_alignment_pairs(train_path)
    val_raw   = load_alignment_pairs(val_path)
    test_raw  = load_alignment_pairs(test_path)

    # Filter to valid pairs
    train_pairs = filter_valid_pairs(train_raw, emb1, emb2)
    val_pairs   = filter_valid_pairs(val_raw,   emb1, emb2)
    test_pairs  = filter_valid_pairs(test_raw,  emb1, emb2)

    print(f"\n[loader] ── Pre-split Results ────────────────")
    print(f"[loader] Train : {len(train_pairs)}")
    print(f"[loader] Val   : {len(val_pairs)}")
    print(f"[loader] Test  : {len(test_pairs)}")
    print(f"[loader] ─────────────────────────────────────")

    return train_pairs, val_pairs, test_pairs