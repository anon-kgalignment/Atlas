#!/usr/bin/env python3
import os, sys
import argparse

# Add local modules folder to path
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from modules.data.loader import load_all
from modules.data.graph import build_merged_graph, build_embedding_matrix
from modules.data.text_encoder import LaBSEEncoder
from modules.train import train_sage


def parse_args():
    parser = argparse.ArgumentParser(description="Entity Alignment")
    parser.add_argument("--embedding_kg1", type=str, required=True, help="Path to KG1 embeddings")
    parser.add_argument("--embedding_kg2", type=str, required=True, help="Path to KG2 embeddings")
    parser.add_argument("--train_links", type=str, required=True, help="Path to train links")
    parser.add_argument("--val_links", type=str, required=True, help="Path to val links")
    parser.add_argument("--test_links", type=str, required=True, help="Path to test links")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--cache", type=str, default=None, help="LaBSE cache file")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    return parser.parse_args()

args = parse_args()
FOLDER_KG1  = args.embedding_kg1
FOLDER_KG2  = args.embedding_kg2
OUTPUT_DIR  = args.output_dir
LABSE_CACHE = args.cache
TRAIN_LINKS = args.train_links
VAL_LINKS   = args.val_links
TEST_LINKS  = args.test_links

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
data = load_all(
    folder_kg1  = FOLDER_KG1,
    folder_kg2  = FOLDER_KG2,
    train_links = TRAIN_LINKS,
    val_links   = VAL_LINKS,
    test_links  = TEST_LINKS,
)

G = build_merged_graph(
    triples1    = data["triples1"],
    triples2    = data["triples2"],
    train_pairs = data["train_pairs"],
    emb1        = data["emb1"],
    emb2        = data["emb2"],
)
E = build_embedding_matrix(G, data["emb1"], data["emb2"])

encoder = LaBSEEncoder(model_name="LaBSE", device=args.device)
P = encoder.encode_graph(
    G          = G,
    names1     = data["names1"],
    names2     = data["names2"],
    batch_size = args.batch_size,
    cache_path = LABSE_CACHE,
)

results, gat, fusion, projector, A = train_sage(
    data             = data,
    G                = G,
    E                = E,
    P                = P,
    output_dir       = OUTPUT_DIR,
    hidden_dim       = 256,
    n_heads          = 4,
    dropout          = 0.1,
    epochs           = args.epochs,
    batch_size       = args.batch_size,
    lr               = args.lr,
    temperature      = 0.07,
    lambda_tc        = 2.0,
    lambda_div       = 0.01,
    expand_every     = 10,
    expand_threshold = 0.95,
    expand_min_drop  = 0.01,
    eval_every       = 1,
    device_str       = args.device,
)

print(f"\nAlignment done. Results saved to {OUTPUT_DIR}")
print(f"EA Hits@1 : {results['test_results']['Hits@1']:.4f}")
print(f"EA MRR    : {results['test_results']['MRR']:.4f}")
print(f"\naligned_kg1.csv and aligned_kg2.csv saved.")
print(f"Now run run_link_prediction.py to evaluate LP.")