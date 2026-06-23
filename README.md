# ATLAS: Adaptive Attribute-Aware Post-Hoc Alignment of Knowledge Graph Embeddings

ATLAS is a **scalable entity alignment framework** for matching entities across independently-trained Knowledge Graph Embedding (KGE) models from different Knowledge Graphs (KGs).

ATLAS combines:
- **Structural embeddings** via Graph Attention Networks (GATs) — one per KG
- **Textual embeddings** via sentence encoders
- **Adaptive fusion** to blend structural and textual information
- **Contrastive learning** for alignment training

Unlike joint-training methods, ATLAS operates **post-hoc**, preserving original KG semantics while learning cross-KG alignments without requiring merged graphs or retraining.

---

## Key Features

**Post-hoc alignment** — No need for joint training or KG merging  
**Scalable** — Separate GAT per KG eliminates cross-KG bottlenecks  
**Multi-modal** — Combines structural (graph) + textual (LaBSE) signals  
**Robust** — Contrastive loss + mutual nearest-neighbor refinement  
**End-to-end** — Entity alignment → link prediction evaluation  


## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- Transformers (for sentence encoder)
- DICE Embedding Framework

### Setup

```bash
cd ATLAS
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Trained KG Embeddings

Each KG directory should contain:

```
kg_embeddings/
- model.pt                   # Trained KGE model checkpoint
- entity_to_idx.p            # Entity → index mapping
- relation_to_idx.p          # Relation → index mapping
- configuration.json         # Model config
```

Use the [DICE Embedding Framework](https://github.com/dice-group/dice-embeddings) to train embeddings (TransE, ComplEx, etc.).

### 2. Triples Files

```
data_dir/
├── train_merged.txt           # All training triples (union of both KGs)
├── test_merged.txt            # All test triples (union of both KGs)
├── rel_triples_1              # DBpedia relation triples (for LP)
├── rel_triples_2              # Wikidata relation triples (for LP)
├── attr_triples_1             # DBpedia attribute triples (names)
└── attr_triples_2             # Wikidata attribute triples (labels)
```

### 3. Alignment Links

```
links_dir/
├── train_links                # Training entity pairs (to align on)
├── valid_links                # Validation entity pairs
└── test_links                 # Test entity pairs (for evaluation)
```

**Sources:**
- [OpenEA Benchmark](https://www.dropbox.com/scl/fi/lo69wjm1f37qiik59kmg8/OpenEA_dataset_v1.1.zip) — DBpedia–Wikidata, multilingual
- [Zenodo – DBpedia-Wikidata](https://zenodo.org/records/7566020) — Full-scale datasets

---

## Usage

### Full Pipeline (Entity Alignment + Link Prediction)

```bash
python3 run_atlas.py \
  --embedding_kg1 <path_to_kg1_embeddings> \
  --embedding_kg2 <path_to_kg2_embeddings> \
  --data_dir <path_to_data> \
  --train_links <path_to_train_links> \
  --val_links <path_to_val_links> \
  --test_links <path_to_test_links> \
  --output_dir <path_to_output> \
  --cache_kg1 <path_to_labse_cache_kg1> \
  --cache_kg2 <path_to_labse_cache_kg2> \
  --ea_epochs 20 \
  --lp_epochs 2 \
  --batch_size 512 \
  --lr 1e-3 \
  --lp_lr 0.001 \
  --device auto
```

### Entity Alignment Only

```bash
python3 run_alignment.py \
  --embedding_kg1 <kg1_embeddings> \
  --embedding_kg2 <kg2_embeddings> \
  --data_dir <data> \
  --train_links <train_links> \
  --val_links <val_links> \
  --test_links <test_links> \
  --output_dir <output>
```

### Link Prediction Only

```bash
python3 run_link_prediction.py \
  --embedding_kg1 <kg1_embeddings> \
  --embedding_kg2 <kg2_embeddings> \
  --aligned_kg1_csv <aligned_kg1.csv> \
  --aligned_kg2_csv <aligned_kg2.csv> \
  --test_triples_path <test_merged.txt> \
  --output_dir <output> \
  --lp_epochs 2
```

---

## Pipeline Steps

When running `run_atlas.py`, the complete pipeline executes:

### Step 1: Load Data
- Loads pre-trained embeddings for both KGs
- Loads alignment links (train/val/test)
- Constructs entity and relation mappings

### Step 2: Build Separate KG Graphs
- Creates two independent graph structures
- **No cross-KG edges** — maintains post-hoc property
- Each KG preserves its original topology

### Step 3: Text Encodings
- Extracts entity names/labels from attribute triples
- Encodes names with a sentence encoder (default: LaBSE, 768-dim multilingual embeddings)
- Caches results for efficiency
- **Note:** The text encoder is modular—you can substitute LaBSE for other models (SBERT, mBERT, XLM-RoBERTa, etc.) by modifying the `Encoder` class in `modules/data/text_encoder.py`

### Step 4: Train ATLAS
- Runs two **Siamese GAT encoders** (one per KG) on separate graphs
- Applies **Adaptive Fusion** to combine:
  - Structural embeddings (from GAT)
  - Textual embeddings (from LaBSE)
- Trains alignment via **contrastive loss (InfoNCE)**
- Refines alignments using **cross-KG nearest-neighbor search**

**Training metrics:** Hits@1, Hits@10, MRR on validation set

### Step 5: Prepare Triples
- Splits merged training triples into:
  - Link Prediction fine-tuning set
  - Link Prediction validation set

### Step 6: Link Prediction
- Injects aligned embeddings into a KGE model
- Fine-tunes using K-vs-all training on merged triples
- Evaluates on test set

**LP metrics:** H@1, H@3, H@10, MRR

---

## Output

Results are saved to `output_dir/`:

```
output_dir/
├── final_results.json         # Combined EA + LP results
├── alignment_results.json     # Entity alignment metrics
├── link_prediction_results.json
│
├── aligned_kg1.csv            # Aligned embeddings for KG1
├── aligned_kg2.csv            # Aligned embeddings for KG2
├── alignment_checkpoint.pt    # Best alignment model weights
├── fusion_model.pt            # Adaptive Fusion module
└── projector_model.pt         # Alignment Projector
```

### Example Output

```json
{
  "entity_alignment": {
    "Hits@1": 0.7234,
    "Hits@10": 0.8456,
    "MRR": 0.7891
  },
  "link_prediction": {
    "H@1": 0.5123,
    "H@3": 0.6234,
    "H@10": 0.7456,
    "MRR": 0.5987
  },
  "best_epoch": 18,
  "mode": "separate_gat_per_kg"
}
```
