# modules/pipeline.py
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# === Model and training ===
from modules.models.train import train_alignment_model 

# === Graph utilities ===
#from modules.graph.build import build_graph_indexes


# === Data loader ===
from modules.data_loader import (
    extract_files_from_directory,
    load_embeddings,
    load_parquet_triples,
    remove_brackets_from_indices,
    build_alignment_dict,
    clean_dict,
    #create_train_test_matrices,
    #normalize_and_scale,
    load_triples_from_files,
    load_triples,
    load_alignment_links, create_train_val_test_matrices_from_links, load_parquet_triples
)


def run_pipeline_for_ckeci(
    directory_1,
    directory_2,
    alignment_dir,
    test_triples_path,
    output_dir,
    triple_paths,
    device="cpu",
):
    # 1) Load embeddings & IDs
    m1, e2i1, r2i1 = extract_files_from_directory(directory_1)
    m2, e2i2, r2i2 = extract_files_from_directory(directory_2)

    ent_df1, rel_df1 = load_embeddings(m1, e2i1, r2i1)
    ent_df2, rel_df2 = load_embeddings(m2, e2i2, r2i2)

    ent1 = remove_brackets_from_indices(ent_df1)
    ent2 = remove_brackets_from_indices(ent_df2)
    rel1 = remove_brackets_from_indices(rel_df1)
    rel2 = remove_brackets_from_indices(rel_df2)
    
    print(f"\n[Loading KG1 triples from {args.train_triples_path_1}]")
    triples_1 = load_triples(args.train_triples_path_1)
    print(f" KG1 triples loaded: {len(triples_1)} triples (first 3): {triples_1[:3]}")
    #G1 = build_graph_indexes(triples_1)

    
    triples_2 = load_triples(args.train_triples_path_2)
    print(f" KG2 triples loaded: {len(triples_2)} triples (first 3): {triples_2[:3]}")
    #G2 = build_graph_indexes(triples_2)


    # 3) Alignment data
    if alignment_dir is not None:
        alignment_dict = clean_dict(build_alignment_dict(alignment_dir))
    else:
        alignment_dict = {}
    #S_train, T_train, S_test, T_test, S_train_keys, T_train_keys, S_test_keys, T_test_keys = (
        #create_train_test_matrices(alignment_dict, ent1, ent2, test_size=0.1)
    #)


    train_links_raw, val_links_raw, test_links_raw = load_alignment_links(args)
    # 3) Alignment data
    train_links = clean_dict(dict(train_links_raw))
    val_links   = clean_dict(dict(val_links_raw))
    test_links  = clean_dict(dict(test_links_raw))


    (S_train, T_train,
     S_val, T_val,
     S_test, T_test,
     S_train_keys, T_train_keys,
     S_val_keys, T_val_keys,
     S_test_keys, T_test_keys) = create_train_val_test_matrices_from_links(
         list(train_links.items()),
        list(val_links.items()),
        list(test_links.items()),
        ent1, ent2
    )
    
    merged_rel = pd.concat([rel1, rel2])
    merged_rel = merged_rel[~merged_rel.index.duplicated(keep="first")]

    #merged_embeddings_full = pd.concat([ent1, ent2])
    #merged_embeddings_full = merged_embeddings_full[~merged_embeddings_full.index.duplicated(keep='first')]
    #print(f" Merged embeddings shape: {merged_embeddings_full.shape}")
    #entities_to_remove = set(train_links.values())
    #print(f" Number of entities to remove (from target side of alignment): {len(entities_to_remove)}")
    #print(f"🔍 Sample entities to remove: {list(entities_to_remove)[:5]}")
    #final_embeddings_df = merged_embeddings_full.drop(labels=entities_to_remove, errors='ignore')
    #print(f"final_embbeddings_df shape: {final_embeddings_df.shape}")
    #sorted_merged_embeddings = final_embeddings_df.sort_index()
    #merged_embeddings_full_sorted = merged_embeddings_full.sort_index()
    
    #merged_embeddings_normalized_without_target, _, _ = normalize_and_scale(sorted_merged_embeddings, reference_data=S_train)
    #merged_embeddings_normalized_with_target, _, _ = normalize_and_scale(merged_embeddings_full_sorted, reference_data=S_train)


    # 4) Triples (used for agents and fine-tuning)
    triples_batch = load_triples_from_files(triple_paths)
    kg1_triples = load_triples_from_files([args.train_triples_path_1])
    kg2_triples = load_triples_from_files([args.train_triples_path_2])
    triples_for_gcn = kg1_triples + kg2_triples

    val_triples, _ = train_test_split(triples_batch, test_size=0.01, random_state=42)

    final_model = train_alignment_model(
        input_dim=256, 
        hidden_dim=256, 
        epochs=2, 
        lr=0.001, 
        S_test_keys=S_test_keys,
        T_test_keys=T_test_keys,
        entity_embeddings1=ent1,
        entity_embeddings2=ent2,
        relation_embeddings=merged_rel,
        output_dir=output_dir,
        triples_batch=triples_for_gcn,
        kg_1=kg1_triples,
        kg_2=kg2_triples,
        device=device,
        #cleaned_alignment_dict=alignment_dict,
        S_train_keys=S_train_keys,
        T_train_keys=T_train_keys,
        S_val_keys=S_val_keys,
        T_val_keys=T_val_keys,
        val_triples=val_triples,
        train_triples=triples_batch,
        directory_1=directory_1,
        test_triples_path=args.test_triples_path
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full NAAS + CKECI pipeline")
    parser.add_argument("--directory_1", required=True)
    parser.add_argument("--directory_2", required=True)
    parser.add_argument("--alignment_dir", default=None)
    parser.add_argument("--train_triples_path_1", required=True, help="Training triples for KG1 (e.g., DBpedia)")
    parser.add_argument("--train_triples_path_2", required=True, help="Training triples for KG2 (e.g., Wikipedia)")
    parser.add_argument("--test_triples_path", required=True)
    parser.add_argument("--train_links", default=None)
    parser.add_argument("--val_links", default=None)
    parser.add_argument("--test_links", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--triple_paths", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_pipeline_for_ckeci(
        directory_1=args.directory_1,
        directory_2=args.directory_2,
        alignment_dir=args.alignment_dir,
        test_triples_path=args.test_triples_path,
        output_dir=args.output_dir,
        triple_paths=args.triple_paths,
        device=args.device,
    )
