import torch
import os, json, random, pickle, logging
import sys
from tqdm import tqdm
import torch.nn.functional as F
import sys, os
from dicee import KGE, intialize_model
import shutil

from modules.data_loader import load_json, sample_encoded_triples
from modules.eval.link_prediction import evaluate_link_prediction_performance
from dicee.static_funcs import get_er_vocab, get_re_vocab


def initialize_models_and_update_embeddings(entity_embeddings, relation_embeddings, output_dir, directory_1):

    """
    Initializes a KGE model using the config in `output_dir` and updates it with
    the given entity and relation embeddings.

    Args:
        entity_embeddings (pd.DataFrame or np.ndarray): Entity embeddings.
        relation_embeddings (pd.DataFrame or np.ndarray): Relation embeddings.
        output_dir (str): Path to the folder containing the configuration and report files.

    Returns:
        model: A KGE model initialized and updated with the provided embeddings.
    """


    def extract_parameters(entity_embeddings, relation_embeddings):
        num_entities = entity_embeddings.shape[0]
        num_relations = relation_embeddings.shape[0]
        embedding_dim = entity_embeddings.shape[1]
        return {
            "num_entities": num_entities,
            "num_relations": num_relations,
            "embedding_dim": embedding_dim
        }
        
    # === Define fine-tune folder ===
    fine_tune_folder = os.path.join(output_dir, "fine_tune")
    os.makedirs(fine_tune_folder, exist_ok=True)

    # === Copy configuration & report from directory_1 ===
    source_config = os.path.join(directory_1, "configuration.json")
    source_report = os.path.join(directory_1, "report.json")

    if os.path.exists(source_config):
        shutil.copy(source_config, fine_tune_folder)
    else:
        raise FileNotFoundError(f"[Error] configuration.json not found in {directory_1}")

    if os.path.exists(source_report):
        shutil.copy(source_report, fine_tune_folder)
    else:
        print("[Warning] report.json not found in source, continuing without it.")


    config_path = os.path.join(fine_tune_folder, "configuration.json")
    report_path = os.path.join(fine_tune_folder, "report.json")

    configs = load_json(config_path)

    
    model_name = configs["model"]  # e.g., "TransE", "ComplEx", etc.
    print(f"Model from config: {model_name}")
    configs["p"] = 0

    if configs["model"] == "Keci":
        configs["q"] = 1
            
    params = extract_parameters(entity_embeddings, relation_embeddings)
    configs["num_entities"] = params["num_entities"]
    configs["num_relations"] = params["num_relations"]
    configs["embedding_dim"] = params["embedding_dim"]

    # Initialize model from config
    model, _ = intialize_model(configs)
    

    if hasattr(model, "q_coefficients"):
        model.q_coefficients.weight.requires_grad = True
        print("q_coefficients enabled")
    else:
        print("Model has no q_coefficients (e.g., TransE) — skipping.")


    # Convert to torch tensors
    entity_tensor = torch.tensor(entity_embeddings, dtype=torch.float32)
    relation_tensor = torch.tensor(relation_embeddings, dtype=torch.float32)

    # Update model's embedding weights
    with torch.no_grad():
        model.entity_embeddings.weight.data = entity_tensor
        model.relation_embeddings.weight.data = relation_tensor
        model.entity_embeddings.weight.requires_grad = False
        model.relation_embeddings.weight.requires_grad = False

    return model


torch.autograd.set_detect_anomaly(True)
def fine_tune_kvsall(
    model,
    aligned_entity_embeddings,
    aligned_relation_embeddings,
    triples_batch,
    val_triples,
    train_triples,
    entity_to_idx,
    relation_to_idx,
    device,
    output_dir,
    batch_size=256,
    epochs=60,
    lr=0.001
):
    if val_triples is None or train_triples is None or len(val_triples) == 0 or len(train_triples) == 0:
        print("[Warning] No validation or training triples provided — skipping fine-tuning.")

    model = model.to(device)
    model.train()
    history = {"epochs": []}

    best_val_mrr = 0
    best_model_state = None
    best_val_metrics = None
    best_epoch = 0

    with torch.no_grad():
        model.entity_embeddings.weight.copy_(torch.tensor(aligned_entity_embeddings, dtype=torch.float32))
        model.relation_embeddings.weight.copy_(torch.tensor(aligned_relation_embeddings, dtype=torch.float32))

    model.entity_embeddings.weight.requires_grad = True
    model.relation_embeddings.weight.requires_grad = True


    if hasattr(model, "q_coefficients"):
        model.q_coefficients = model.q_coefficients.to(device)
        model.q_coefficients.weight.requires_grad = True
        print("q_coefficients initial value:", model.q_coefficients.weight.detach().cpu().numpy())
    else:
        print("Skipping q_coefficients (model does not use them)")


    model.train()
    random.shuffle(triples_batch)
    n = len(triples_batch)
    batch_1 = triples_batch[:n//3]
    batch_2 = triples_batch[n//3:2*n//3]
    batch_3 = triples_batch[2*n//3:]
    

    print("Trainable params now:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {tuple(param.shape)}")

        
    #  Confirm requires_grad is set correctly
    for name, param in model.named_parameters():
        print(f" {name}: requires_grad={param.requires_grad}")

    param_groups = [
        {"params": [model.entity_embeddings.weight, model.relation_embeddings.weight], "lr": lr},
    ]
    if hasattr(model, "q_coefficients"):
        param_groups.append({"params": model.q_coefficients.parameters(), "lr": lr * 5.0})
    optimizer = torch.optim.Adam(param_groups, weight_decay=0.0)
    

    sample_new_triples = False
    
    sample_size = min(len(triples_batch), 60000)

    encoded_triples = sample_encoded_triples(triples_batch, entity_to_idx, relation_to_idx, sample_size=sample_size)
    for epoch in range(epochs):
        if epoch < 20:
            current_batch = batch_1
        elif epoch < 40:
            current_batch = batch_2
        else:
            current_batch = batch_3
        if epoch==20:
            sample_new_triples=True
        elif epoch==40:
            sample_new_triples=True
        if sample_new_triples:
            encoded_triples = sample_encoded_triples(current_batch, entity_to_idx, relation_to_idx, sample_size=60000)
            sample_new_triples=False


        total_loss = 0.0
        random.shuffle(encoded_triples)

        for i in range(0, len(encoded_triples), batch_size):
            batch = encoded_triples[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)  # [B, 3]

            # produces real-valued scores (logits) for various entities. These continuous scores are akin to what a "belief function" (bθ) might output, representing the model's confidence or degree of belief for each entity
            scores = model.forward_k_vs_all(batch_tensor)  # [B, num_entities]

            B, N = scores.shape
            rows = torch.arange(B, device=device)
            true_tails = batch_tensor[:, 2]
            k = 50  # or 100

            # pick top-k hard negatives per row (exclude true tail)
            with torch.no_grad():
                s = scores.detach().clone()
                s[rows, true_tails] = float('-inf')
                hard_negs = torch.topk(s, k, dim=1).indices  # [B, k]

            # gather logits for positive + k negatives -> [B, k+1]
            pos = scores[rows, true_tails].unsqueeze(1)  # [B, 1]
            negs = scores.gather(1, hard_negs)           # [B, k]
            logits = torch.cat([pos, negs], dim=1)       # [B, k+1]

            # class 0 is the positive
            targets_ce = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, targets_ce)

       
            #  Confirm loss is connected to model
            assert loss.requires_grad, "Loss has no grad — check model/scoring logic!"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs}, KvsAll BCE Loss: {total_loss:.4f}")
        avg_loss = total_loss / (len(encoded_triples) / batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Avg BCE Loss: {avg_loss:.4f}")
        
        if hasattr(model, "q_coefficients"):
            print("q_coefficients now:", model.q_coefficients.weight.detach().cpu().numpy())
            
         # === Save fine-tuned model directly inside output_dir/fine_tune ===
        fine_tune_folder = os.path.join(output_dir, "fine_tune")
        os.makedirs(fine_tune_folder, exist_ok=True)
                    
        torch.save(model.state_dict(), os.path.join(fine_tune_folder, "model.pt"))
        print(f" Fine-tuned model saved at {os.path.join(fine_tune_folder, 'model.pt')}")
        
        print("Evaluating on validation triples after fine-tuning...")
    
        
        def get_entity_coverage(triples):
            heads = {h for (h, _, _) in triples}
            tails = {t for (_, _, t) in triples}
            return heads.union(tails)

                
        with open(os.path.join(fine_tune_folder, "entity_to_idx.p"), "wb") as f:
            pickle.dump(entity_to_idx, f)

        with open(os.path.join(fine_tune_folder, "relation_to_idx.p"), "wb") as f:
            pickle.dump(relation_to_idx, f)
            
            
        train_triples_1000 = train_triples[:3000]

        train_er_vocab_1000 = get_er_vocab(train_triples_1000)
        train_re_vocab_1000 = get_re_vocab(train_triples_1000)
                    
                
        train_er_vocab = get_er_vocab(train_triples)
        train_re_vocab = get_re_vocab(train_triples)
            
        val_triples_1000 = val_triples[:3000]

        val_er_vocab_1000 = get_er_vocab(val_triples_1000)
        val_re_vocab_1000 = get_re_vocab(val_triples_1000)

                
       
        # Load the final model
        finetuned_model = KGE(path= fine_tune_folder)
    
        # Evaluate only on the 1000 triples
        train_metrics = evaluate_link_prediction_performance(finetuned_model, train_triples_1000, train_er_vocab_1000, train_re_vocab_1000)
        print(f" Finetuned Model Performance on first 1000 triples train set:\n{train_metrics}")

        val_metrics = evaluate_link_prediction_performance(finetuned_model, val_triples_1000, val_er_vocab_1000, val_re_vocab_1000)

        # capture q coefficient(s)
        if hasattr(model, "q_coefficients"):
            q_vec = model.q_coefficients.weight.detach().cpu().flatten().tolist()
        else:
            q_vec = None

            
        history["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": float(total_loss),
            "train_metrics": {k: float(v) for k, v in train_metrics.items()},
            "val_metrics":   {k: float(v) for k, v in val_metrics.items()},
            "q": q_vec 
        })

        
        if val_metrics["MRR"] > best_val_mrr:
            best_val_mrr = val_metrics["MRR"]
            best_model_state = model.state_dict()
            best_val_metrics = val_metrics
            best_epoch = epoch + 1
            print(f" New best model at epoch {best_epoch} with val MRR: {best_val_mrr:.4f}")


        print("Training Performance (MRR / Hits@K):")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.4f}")
            

        print("Validation Performance (MRR / Hits@K):")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")
            
                
    # Save evaluation metrics
    with open(os.path.join(fine_tune_folder, "fine_tune_metrics.json"), "w") as mf:
        json.dump({"train": train_metrics, "val": val_metrics}, mf, indent=2)

    with open(os.path.join(fine_tune_folder, "fine_tune_history.json"), "w") as mf:
        json.dump(history, mf, indent=2)

    # Save latest checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(fine_tune_folder, "fine_tuned_checkpoint.pth"))

    # Save best model (also into fine_tune/)
    if best_model_state is not None:
        torch.save(best_model_state, os.path.join(fine_tune_folder, "model.pt"))
        with open(os.path.join(fine_tune_folder, "best_model_metadata.json"), "w") as f:
            json.dump({
                "best_epoch": best_epoch,
                "best_val_mrr": best_val_mrr,
                "val_metrics": best_val_metrics
            }, f, indent=2)
        print(f" Best model saved at epoch {best_epoch} in {fine_tune_folder}")
        
   