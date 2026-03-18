#  ASTRA: Adaptive Structure-Aware Post-Hoc Alignment of Knowledge Graph Embeddings

ASTRA is a **post-hoc alignment framework** for aligning entity embeddings from independently trained Knowledge Graph Embedding (KGE) models.

It enables embeddings from different knowledge graphs (KGs) to be used for downstream tasks—such as **entity alignment** and **link prediction**—without requiring joint training or KG merging.

---

#  Motivation

Traditional alignment methods (e.g., **MTransE, BootEA, KDCoE**) have several limitations:

- Require **joint training**
- Need **merged graphs**
- Do **not scale well** to large KGs

---

### ASTRA's Approach

ASTRA follows a different paradigm:

- Align embeddings **after training (post-hoc)**
- Preserve original semantic information
- Inject **graph structure via R-GCN**
- Learn **non-linear alignment**

---

# Pipeline Overview

```text
1. Load pretrained embeddings (KG1, KG2)
2. Load triples (txt / parquet)
3. Build merged graph
4. Compute RGCN embeddings
5. Fuse embeddings
6. Train alignment model
7. Evaluate entity alignment
8. Inject aligned embeddings into KGE
9. Fine-tune using sampled triples
10. Evaluate link prediction
```
---


## Code and Data

Embeddings: Trained using the [DICE Embedding Framework](https://github.com/dice-group/dice-embeddings)

Each embedding directory should contain:

```text
model.pt
entity_to_idx.p
relation_to_idx.p
configuration.json
```

Datasets:
[OpenEA benchmark](https://www.dropbox.com/scl/fi/lo69wjm1f37qiik59kmg8/OpenEA_dataset_v1.1.zip)
[Zenodo – DBpedia-Wikidata](https://zenodo.org/records/7566020)


### Required Files

#### Triples

```text
rel_triples_1_train.txt
rel_triples_2_train.txt
rel_triples_test_merged.txt (merged test triples from both dataset)
rel_triples_train_merged.txt (merged train triples from both dataset)
```

#### Alignment Links

```text
train_links
valid_links
test_links
```

---

##  Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

##  Installation Steps

1. **Train KGE embeddings**
   Train entity and relation embeddings using the **DICE Embedding Framework** (or any compatible KGE model such as TransE, ComplEx, etc.).

2. **Prepare datasets**
   Download or prepare:

   * Knowledge graph triples (train/test)
   * Alignment links (train / validation / test)

   Supported sources include:

   * OpenEA benchmark datasets
   * DBpedia–Wikidata datasets

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Run Pipeline

```bash
python3 -m modules.pipeline \
  --directory_1 <KG1_embeddings> \
  --directory_2 <KG2_embeddings> \
  --train_triples_path_1 <KG1_train_triples> \
  --train_triples_path_2 <KG2_train_triples> \
  --test_triples_path <merged_test_triples> \
  --triple_paths <merged_train_triples> \
  --train_links <train_links> \
  --val_links <validation_links> \
  --test_links <test_links> \
  --output_dir <output_directory>
```

---

##  What the Pipeline Does

The pipeline performs the following steps:

1. Loads pretrained embeddings for both KGs
2. Loads triples and alignment links
3. Builds a **merged graph structure**
4. Computes **R-GCN structural embeddings**
5. Applies **adaptive fusion** (structure + base embeddings)
6. Trains the **alignment model**
7. Evaluates **entity alignment (Hits@k, MRR)**
8. Injects aligned embeddings into a KGE model
9. Performs **fine-tuning (KvsAll training)**
10. Evaluates **link prediction performance**

--

## Output

Results are saved in:

```text
output_dir/
│
├── alignment_results.json
├── link_prediction_results.json
│
├── aligned_embeddings/        # aligned entity embeddings
├── fine_tuned_model/          # final fine-tuned KGE model
```

---

##  Notes

* No KG merging required
* No KGE retraining required
* Sampling is **random and dynamic**
* Works with multiple KGE models
