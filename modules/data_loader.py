# modules/data_loader.py
import argparse
import logging
import os
import pandas as pd
import random
from rdflib import Graph
from urllib.parse import unquote
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import json
import torch.nn.functional as F

def sample_encoded_triples(triples_batch, entity_to_idx, relation_to_idx, sample_size=60000):
    sampled_triples = random.sample(triples_batch, min(sample_size, len(triples_batch)))
    encoded = []
    for h, r, t in sampled_triples:
        h_clean = clean_uri(h)
        r_clean = clean_uri(r)
        t_clean = clean_uri(t)
        if h_clean in entity_to_idx and r_clean in relation_to_idx and t_clean in entity_to_idx:
            encoded.append((entity_to_idx[h_clean], relation_to_idx[r_clean], entity_to_idx[t_clean]))
    return encoded

def load_json(p: str) -> dict:
        with open(p, 'r') as r:
            args = json.load(r)
        return args


def extract_files_from_directory(directory):
    """Retrieve required files (model, entity_to_idx, relation_to_idx) from a directory."""
    model_path = os.path.join(directory, "model.pt")

    # Possible file formats for entity_to_idx and relation_to_idx
    entity_to_id_options = [
        os.path.join(directory, "entity_to_idx.p"),
        os.path.join(directory, "entity_to_idx"),
        os.path.join(directory, "entity_to_idx.csv")
    ]
        
    relation_to_id_options = [
        os.path.join(directory, "relation_to_idx.p"),
        os.path.join(directory, "relation_to_idx"),
        os.path.join(directory, "relation_to_idx.csv")
    ]

    # Check for existing entity_to_idx file
    entity_to_id_path = next((path for path in entity_to_id_options if os.path.exists(path)), None)
    if entity_to_id_path is None:
        logging.error(f"Entity-to-ID file not found in {directory}. Checked: {entity_to_id_options}")
        raise FileNotFoundError(f"Missing entity_to_idx file in {directory}.")

    # Check for existing relation_to_idx file
    relation_to_id_path = next((path for path in relation_to_id_options if os.path.exists(path)), None)
    if relation_to_id_path is None:
        logging.error(f"Relation-to-ID file not found in {directory}. Checked: {relation_to_id_options}")
        raise FileNotFoundError(f"Missing relation_to_idx file in {directory}.")

    logging.debug(f"Files found: Model={model_path}, Entity-to-ID={entity_to_id_path}, Relation-to-ID={relation_to_id_path}")

    return model_path, entity_to_id_path, relation_to_id_path
    

def load_embeddings(model_path, entity_to_id_path, relation_to_id_path):
        """Load embeddings and mappings for entities and relations."""
            
        logging.info(f"Loading model weights from: {model_path}")
        model_weights = torch.load(model_path, map_location='cpu', weights_only=True)

        # Load entity and relation embeddings
        entity_embeddings = model_weights['entity_embeddings.weight'].cpu().detach().numpy()
        relation_embeddings = model_weights['relation_embeddings.weight'].cpu().detach().numpy()

        logging.info(f"Entity embeddings shape: {entity_embeddings.shape}")
        logging.info(f"Relation embeddings shape: {relation_embeddings.shape}")

        # Handle entity-to-ID mapping
        try:
            entity_df = pd.read_csv(entity_to_id_path)
            entity_to_id = dict(zip(entity_df.index, entity_df["entity"]))
            logging.info(f"Successfully loaded entity mappings from Parquet (without extension): {entity_to_id_path}")
            
        except Exception as e:
            raise ValueError(f"Could not load entity-to-ID mapping from {entity_to_id_path}: {str(e)}")

        # Handle relation-to-ID mapping similarly
        try:
            relation_df = pd.read_csv(relation_to_id_path)
            relation_to_id = dict(zip(relation_df.index, relation_df["relation"]))
            logging.info(f"Successfully loaded relation mappings from Parquet (without extension): {relation_to_id_path}")

        except Exception as e:
            raise ValueError(f"Could not load relation-to-ID mapping from {relation_to_id_path}: {str(e)}")

        # Match embeddings to URIs using entity index
        sorted_entities = [entity_to_id[i] for i in range(len(entity_embeddings))]
        sorted_relations = [relation_to_id[i] for i in range(len(relation_embeddings))]

        # Ensure lengths match
        if len(entity_embeddings) != len(sorted_entities):
            logging.warning(f"Mismatch: {len(entity_embeddings)} entity embeddings vs. {len(sorted_entities)} entities. Fixing it...")
            sorted_entities = sorted_entities[:len(entity_embeddings)]
            
        if len(relation_embeddings) != len(sorted_relations):
            logging.warning(f"Mismatch: {len(relation_embeddings)} relation embeddings vs. {len(sorted_relations)} relations. Fixing it...")
            sorted_relations = sorted_relations[:len(relation_embeddings)]

        # Create DataFrames
        entity_embeddings_df = pd.DataFrame(entity_embeddings, index=sorted_entities)
        relation_embeddings_df = pd.DataFrame(relation_embeddings, index=sorted_relations)

        return entity_embeddings_df, relation_embeddings_df
    

def remove_brackets_from_indices(embeddings_df):
    """Remove < and > from each index in the DataFrame."""
    cleaned_index = [uri.strip('<>') for uri in embeddings_df.index]
    embeddings_df.index = cleaned_index
    return embeddings_df


def clean_uri(uri):
    """Normalize URIs: remove angle brackets, quotes, decode escapes, strip whitespace."""
    if uri is None:
        return ""
    u = str(uri).strip()
    u = u.replace("<<", "").replace(">>", "").replace("<", "").replace(">", "")
    u = u.strip().strip('"')  # remove quotes if any
    u = unquote(u)            # decode %20, %3A etc.
    # Optional: normalize scheme if you have mix of http/https
    # u = u.replace("https://", "http://", 1)
    return u

    
def build_alignment_dict(path):
    """
    Build an alignment dictionary from a single alignment file OR a folder of files.

    Accepts:
        - a folder containing multiple .txt/.nt/.ttl files
        - a single file path

    Returns:
        dict: {subject_uri : object_uri}
    """
    alignment_dict = {}

    # -------------------------
    # CASE 1: Input is a FILE
    # -------------------------
    if os.path.isfile(path):
        file_paths = [path]
    # -------------------------
    # CASE 2: Input is a FOLDER
    # -------------------------
    elif os.path.isdir(path):
        if not os.listdir(path):
            logging.warning(f"Alignment folder '{path}' is empty. Skipping.")
            return alignment_dict
        file_paths = [os.path.join(path, f) for f in os.listdir(path)]
    else:
        logging.error(f"Path '{path}' does not exist.")
        return alignment_dict

    # Process each file
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()

        try:
            # -------- RDF files (.nt, .ttl) --------
            if ext in ['.nt', '.ttl']:
                g = Graph()
                g.parse(file_path, format='nt' if ext == '.nt' else 'ttl')

                for subj, pred, obj in g:
                    subj = clean_uri(str(subj))
                    pred = clean_uri(str(pred))
                    obj = clean_uri(str(obj))

                    if "sameAs" in pred:
                        alignment_dict[subj] = obj

            # -------- TXT/TSV files --------
            elif ext in ['', '.txt']:
                with open(file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            alignment_dict[clean_uri(parts[0])] = clean_uri(parts[1])
                        elif len(parts) == 4 and "sameAs" in parts[1]:
                            alignment_dict[clean_uri(parts[0])] = clean_uri(parts[2])
                        else:
                            logging.warning(f"Skipping bad line: {line.strip()}")

            else:
                logging.warning(f"Unsupported file type: {file_path}")

        except Exception as e:
            logging.error(f"Error while reading {file_path}: {e}")

    return alignment_dict



def clean_dict(input_dict):
    """
    Cleans a dictionary by removing angle brackets and any trailing characters like '> .'
    fro both keys and values.
        """
    return {
        k.strip('<>'): v.strip('<>') if isinstance(v, str) else v
        for k, v in input_dict.items()
    }
    
    
def create_train_test_matrices(alignment_dict, entity_embeddings1, entity_embeddings2, test_size=0.1):
    """Generates train and test matrices based on alignment dictionary and keeps entity URIs."""

    # Filter only the valid alignment pairs
    filtered_alignment_dict = {
        k: v for k, v in alignment_dict.items() if k in entity_embeddings1.index and v in entity_embeddings2.index
    }

    if not filtered_alignment_dict:
        logging.warning("No valid entries found in the alignment dictionary. Skipping train-test split.")
        return None, None, None, None, None, None, None, None

    # Perform train-test split on entity names
    train_ents, test_ents = train_test_split(
        list(filtered_alignment_dict.keys()), test_size=test_size, random_state=42)

    # Create lists of aligned entities
    S_train_keys = train_ents
    T_train_keys = [filtered_alignment_dict[e] for e in train_ents]

    S_test_keys = test_ents
    T_test_keys = [filtered_alignment_dict[e] for e in test_ents]

    # Retrieve embeddings
    S_train = entity_embeddings1.loc[S_train_keys].values
    T_train = entity_embeddings2.loc[T_train_keys].values
    S_test = entity_embeddings1.loc[S_test_keys].values if S_test_keys else None
    T_test = entity_embeddings2.loc[T_test_keys].values if S_test_keys else None

    #  Return embeddings and keys
    return S_train, T_train, S_test, T_test, S_train_keys, T_train_keys, S_test_keys, T_test_keys


import logging

def load_alignment_links(args):
    """
    Loads train/val/test link pairs from *single files*.
    Falls back to alignment_dir only if none provided.
    """

    def read_alignment_file(path):
        """Load (src, tgt) pairs from a file."""
        if not path or not os.path.exists(path):
            return []

        d = build_alignment_dict(path)  # already handles .txt, .nt, .ttl
        return list(d.items())

    # --- 1) Read direct file paths (your case!) ---
    train_links = read_alignment_file(args.train_links)
    val_links   = read_alignment_file(args.val_links)
    test_links  = read_alignment_file(args.test_links)

    # --- 2) fallback to alignment_dir ---
    if not train_links and getattr(args, "alignment_dir", None):
        print(f"Using alignment_dir={args.alignment_dir}")
        d = build_alignment_dict(args.alignment_dir)
        train_links = list(d.items())

    return train_links, val_links, test_links

def create_train_val_test_matrices_from_links(train_links, val_links, test_links,
                                              entity_embeddings1, entity_embeddings2):
    """Generates train, validation, and test matrices using pre-defined alignment links."""

    def filter_links(links, name):
        filtered = []
        skipped = 0

        print(f"\n=== Debug: First 5 items in {name}_links ===")
        for i, link in enumerate(links[:5]):
            print(f"{i}: {link} (type: {type(link)})")
            if isinstance(link, (list, tuple)):
                print(f"    Length: {len(link)}")
            else:
                print(f"    WARNING: Not a list or tuple!")

        for link in links:
            if len(link) != 2:
                skipped += 1
                continue
            e1, e2 = link
            if e1 in entity_embeddings1.index and e2 in entity_embeddings2.index:
                filtered.append((e1, e2))
        if skipped > 0:
            logging.warning(f"{skipped} malformed links were skipped in {name}_links.")
        logging.info(f"{len(filtered)} valid links retained in {name}_links.")
        return filtered

    # --- Filter all link sets ---
    filtered_train_links = filter_links(train_links, "train")
    filtered_val_links   = filter_links(val_links, "val")
    filtered_test_links  = filter_links(test_links, "test")

    # --- Train embeddings ---
    S_train = entity_embeddings1.loc[[e1 for e1, _ in filtered_train_links]].values if filtered_train_links else None
    T_train = entity_embeddings2.loc[[e2 for _, e2 in filtered_train_links]].values if filtered_train_links else None
    S_train_keys = [e1 for e1, _ in filtered_train_links]
    T_train_keys = [e2 for _, e2 in filtered_train_links]

    # --- Validation embeddings ---
    S_val = entity_embeddings1.loc[[e1 for e1, _ in filtered_val_links]].values if filtered_val_links else None
    T_val = entity_embeddings2.loc[[e2 for _, e2 in filtered_val_links]].values if filtered_val_links else None
    S_val_keys = [e1 for e1, _ in filtered_val_links]
    T_val_keys = [e2 for _, e2 in filtered_val_links]

    # --- Test embeddings ---
    S_test = entity_embeddings1.loc[[e1 for e1, _ in filtered_test_links]].values if filtered_test_links else None
    T_test = entity_embeddings2.loc[[e2 for _, e2 in filtered_test_links]].values if filtered_test_links else None
    S_test_keys = [e1 for e1, _ in filtered_test_links]
    T_test_keys = [e2 for _, e2 in filtered_test_links]

    # --- Return all (now includes validation keys!) ---
    return (
        S_train, T_train,
        S_val, T_val,
        S_test, T_test,
        S_train_keys, T_train_keys,
        S_val_keys, T_val_keys,    
        S_test_keys, T_test_keys
    )

    
def lang_counts(index):
    idx = list(index)
    en = sum('://dbpedia.org/resource/' in u and '://fr.dbpedia.org/' not in u for u in idx)
    de = sum('://fr.dbpedia.org/resource/' in u for u in idx)
    other = len(idx) - en - de
    return en, de, other


def normalize_and_scale(data, reference_data=None):
        """
        Normalize and scale data using its mean and standard deviation, or those of a reference dataset.
        
        Parameters:
        - data: ndarray, the data to be normalized.
        - reference_data: ndarray, the reference data used for calculating mean and scale. If None, uses `data` itself.
        
        Returns:
        - normalized_data: ndarray, the normalized data.
        - mean: ndarray, the mean used for normalization.
        - scale: float, the scale used for normalization.
        """
        if reference_data is None:
            reference_data = data

        mean = reference_data.mean(axis=0)
        scale = np.sqrt(((reference_data - mean) ** 2).sum() / reference_data.shape[0])
        normalized_data = (data - mean) / scale

        return normalized_data, mean, scale
    
    
def normalize_embedding_space(S, T):
    # ONLY per-vector normalization
    S_norm = F.normalize(S, p=2, dim=1)
    T_norm = F.normalize(T, p=2, dim=1)

    return S_norm, T_norm

    

def load_triples(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, engine="fastparquet")
        if {"subject", "relation", "object"}.issubset(df.columns):
            return list(zip(df["subject"], df["relation"], df["object"]))
        else:
            raise ValueError(f"Unexpected columns in {path}: {df.columns}")
    elif path.endswith(".txt"):
        with open(path, "r") as f:
            return [tuple(line.strip().split()) for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported triples file type: {path}")

    
def load_triples_from_files(file_paths):
    """
    Load ALL triples from one or more files (txt or parquet), with no sampling.

    Args:
        file_paths (list): List of file paths to load triples from.

    Returns:
        List of (head, relation, tail) triples as strings.
    """
    triples = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.warning(f"Triple file not found: {file_path}")
            continue

        try:
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path, engine="fastparquet")

                for _, row in df.iterrows():
                    triples.append((str(row[0]), str(row[1]), str(row[2])))

            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            triples.append((parts[0], parts[1], parts[2]))

        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

    return triples

        
def load_parquet_triples(path):
    """
    Load triples from a parquet file with 3 columns: head, relation, tail.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Triple file does not exist: {path}")

    df = pd.read_parquet(path, engine="fastparquet")
    if df.shape[1] < 3:
        raise ValueError(f"Triple parquet {path} does not have 3 columns")

    triples = [(str(row[0]), str(row[1]), str(row[2])) for _, row in df.iterrows()]
    return triples
