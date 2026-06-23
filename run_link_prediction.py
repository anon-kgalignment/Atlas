#!/usr/bin/env python3
import os
import sys
import argparse
import random
import numpy as np
import torch
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Add local modules folder to path
sys.path.insert(0, os.path.dirname(__file__))

from modules.eval.link_prediction import run_link_prediction_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Link Prediction")
    parser.add_argument("--embedding_kg1", type=str, required=True, help="Path to KG1 embeddings")
    parser.add_argument("--embedding_kg2", type=str, required=True, help="Path to KG2 embeddings")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--alignment_dir", type=str, required=True, help="Path with aligned_kg1.csv and aligned_kg2.csv")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Fine-tuning learning rate")
    return parser.parse_args()

args = parse_args()
FOLDER_KG1 = args.embedding_kg1
FOLDER_KG2 = args.embedding_kg2
OUTPUT_DIR = args.alignment_dir
DATA_DIR   = args.data_dir

TRAIN_TRIPLES = os.path.join(DATA_DIR, "train_merged.txt")
TEST_TRIPLES  = os.path.join(DATA_DIR, "test_merged.txt")
REL_TRIPLES_1 = os.path.join(DATA_DIR, "rel_triples_1")
REL_TRIPLES_2 = os.path.join(DATA_DIR, "rel_triples_2")

aligned_kg1 = os.path.join(OUTPUT_DIR, "aligned_kg1.csv")
aligned_kg2 = os.path.join(OUTPUT_DIR, "aligned_kg2.csv")

if not os.path.exists(aligned_kg1):
    print("ERROR: aligned_kg1.csv not found.")
    print("Run run_alignment.py first.")
    sys.exit(1)

print(f"Using existing aligned embeddings from {OUTPUT_DIR}")

def load_triples_tsv(path):
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["subject", "relation", "object"],
        dtype=str, engine="python",
    ).dropna()
    return df.values.tolist()

all_train = load_triples_tsv(TRAIN_TRIPLES)
random.seed(42)
random.shuffle(all_train)
n_val            = max(1000, int(len(all_train) * 0.1))
val_triples_lp   = all_train[:n_val]
train_triples_lp = all_train[n_val:]

print(f"Train triples : {len(train_triples_lp)}")
print(f"Val triples   : {len(val_triples_lp)}")

FINE_TUNE_EPOCHS = args.epochs
FINE_TUNE_LR     = args.lr

lp_results = run_link_prediction_pipeline(
    aligned_kg1_csv    = aligned_kg1,
    aligned_kg2_csv    = aligned_kg2,
    folder_kg1         = FOLDER_KG1,
    folder_kg2         = FOLDER_KG2,
    train_triples      = train_triples_lp,
    val_triples        = val_triples_lp,
    test_triples_path  = TEST_TRIPLES,
    output_dir         = OUTPUT_DIR,
    fine_tune_epochs   = FINE_TUNE_EPOCHS,
    fine_tune_lr       = FINE_TUNE_LR,
    rel_triples_1_path = REL_TRIPLES_1,
    rel_triples_2_path = REL_TRIPLES_2,
)

print(f"\nLink Prediction Results:")
print(f"  H@1  : {lp_results['H@1']:.4f}")
print(f"  H@3  : {lp_results['H@3']:.4f}")
print(f"  H@10 : {lp_results['H@10']:.4f}")
print(f"  MRR  : {lp_results['MRR']:.4f}")
