"""
Trained Bi-Encoder Baseline (unified).
Trains a bi-encoder with supervised contrastive loss (InfoNCE) on the SAME
training data as our method, using the SAME DataLoader pipeline.
Supports any HuggingFace encoder:
  - microsoft/deberta-v3-base
  - roberta-base (R-SupCon paradigm)

Pipeline: Train encoder → Build KB index → Retrieve → Evaluate.
No GNN, no table structure, no hypergraph — pure text matching.
"""
import argparse
import os
import sys
import json
import random

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"
import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from eval_utils import load_eval_samples, evaluate_predictions, save_results, KB_PATH
from data.dataset import TableAlignmentDataset
from torch_geometric.loader import DataLoader


class BiEncoderModel(nn.Module):
    """Generic bi-encoder: any HuggingFace model + projection head."""

    def __init__(self, model_name="roberta-base", proj_dim=256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode(self, texts, batch_size=16):
        # DeBERTa-v2 SDPA / gradient_checkpointing internals can return fp16
        # tensors on H100 even without an outer autocast. Force the same dtype
        # as the proj MLP weight to avoid Half/Float mismatch.
        target_dtype = self.proj[0].weight.dtype
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(next(self.parameters()).device)
            outputs = self.encoder(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :].to(target_dtype)
            proj_embeds = self.proj(cls_embeds)
            all_embeds.append(F.normalize(proj_embeds, p=2, dim=-1))
        return torch.cat(all_embeds, dim=0)


def extract_training_pairs(graph, kb_entities, ent_id_to_idx):
    """Extract (cell_text, positive_entity_idx) pairs from a graph batch."""
    labeled_indices = graph.labeled_indices
    target_ent_ids = graph.target_ent_ids

    if isinstance(target_ent_ids[0], (list, tuple)):
        target_ent_ids = [eid for sublist in target_ent_ids for eid in sublist]

    x_text = graph.x_text
    if x_text and isinstance(x_text[0], list):
        x_text = [t for group in x_text for t in group]

    pairs = []
    for i, eid in enumerate(target_ent_ids):
        if i >= labeled_indices.numel():
            break
        node_idx = labeled_indices[i].item()
        if node_idx >= len(x_text):
            continue

        eid_str = str(eid).strip()
        eid_clean = eid_str.split("/")[-1]
        match_idx = ent_id_to_idx.get(eid_str) or ent_id_to_idx.get(eid_clean) or \
                    ent_id_to_idx.get(eid_str.lower()) or ent_id_to_idx.get(eid_clean.lower())
        if match_idx is not None:
            pairs.append((x_text[node_idx], match_idx))

    return pairs


def infonce_loss(query_embeds, pos_embeds, neg_embeds, temperature):
    """InfoNCE contrastive loss."""
    pos_sim = torch.sum(query_embeds * pos_embeds, dim=-1, keepdim=True) / temperature
    neg_sim = torch.matmul(query_embeds, neg_embeds.t()) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def main():
    parser = argparse.ArgumentParser(description="Trained Bi-Encoder Baseline")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model: microsoft/deberta-v3-base, roberta-base")
    parser.add_argument("--tag", type=str, default="",
                        help="Display name for results (auto-generated if empty)")
    parser.add_argument("--train_data", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/datasets/wdc_lspm_sampled/train"))
    parser.add_argument("--kb_path", type=str, default=KB_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_negatives", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load checkpoint and evaluate only")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint for eval_only mode")
    args = parser.parse_args()

    if not args.tag:
        short = args.model_name.split("/")[-1]
        if "roberta" in args.model_name.lower():
            args.tag = "DeBERTa+R-SupCon paradigm (controlled)"
        else:
            args.tag = f"DeBERTa Bi-Encoder ({short}, trained)"

    safe_name = args.model_name.replace("/", "_").replace("-", "_")

    print("=" * 60)
    print(f"Trained Bi-Encoder Baseline: {args.tag}")
    print(f"Model: {args.model_name}, Epochs: {args.epochs}")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BiEncoderModel(args.model_name).to(device)

    kb_entities = []
    with open(args.kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_entities.append(json.loads(line))
    kb_names = [e["name"] for e in kb_entities]
    num_kb = len(kb_entities)
    print(f"[KB] Loaded {num_kb} entities")

    ent_id_to_idx = {}
    for i, e in enumerate(kb_entities):
        raw_id = str(e['id'])
        ent_id_to_idx[raw_id] = i
        ent_id_to_idx[raw_id.split("/")[-1].strip()] = i
        ent_id_to_idx[raw_id.lower()] = i
        ent_id_to_idx[raw_id.split("/")[-1].strip().lower()] = i

    ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/biencoder_{safe_name}.pt")

    if args.eval_only:
        load_path = args.checkpoint or ckpt_path
        print(f"\n[eval_only] Loading checkpoint: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
        # Checkpoint may be saved under AMP (encoder fp16, proj fp32 → mismatch
        # at eval time without autocast). Force fp32 across the whole model.
        model.float()
        print("  Checkpoint loaded (dtype unified to fp32), skipping training.")
    else:
        # ===== Training (same DataLoader as main experiment) =====
        print(f"\n[1/3] Training {args.tag}...")
        train_dataset = TableAlignmentDataset(root=args.train_data)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=4,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss, valid_steps = 0.0, 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                pairs = extract_training_pairs(batch, kb_entities, ent_id_to_idx)
                if len(pairs) < 2:
                    continue

                cell_texts = [p[0] for p in pairs]
                pos_indices = [p[1] for p in pairs]
                pos_names = [kb_names[idx] for idx in pos_indices]

                neg_indices = []
                pos_set = set(pos_indices)
                while len(neg_indices) < args.num_negatives:
                    idx = random.randint(0, num_kb - 1)
                    if idx not in pos_set:
                        neg_indices.append(idx)
                neg_names = [kb_names[i] for i in neg_indices]

                query_embeds = model.encode(cell_texts)
                pos_embeds = model.encode(pos_names)
                neg_embeds = model.encode(neg_names)

                loss = infonce_loss(query_embeds, pos_embeds, neg_embeds, model.temperature)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                valid_steps += 1

            avg_loss = total_loss / max(1, valid_steps)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"Epoch {epoch+1} | loss={avg_loss:.4f} ✓ best → saved")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1} | loss={avg_loss:.4f} ✗ no improve ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Loaded best checkpoint: {ckpt_path}")

    # ===== Build KB index =====
    print(f"\n[2/3] Building KB index with trained {args.tag}...")
    model.eval()

    all_kb_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, num_kb, 256), desc="Encoding KB"):
            batch = kb_names[i:i+256]
            embeds = model.encode(batch, batch_size=256)
            all_kb_embeds.append(embeds.cpu())
    kb_embeddings = torch.cat(all_kb_embeds, dim=0)

    # ===== Evaluation =====
    print("\n[3/3] Evaluating...")
    samples = load_eval_samples()
    cell_texts = [s["cell_text"] for s in samples]

    all_query_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(cell_texts), 256), desc="Encoding queries"):
            batch = cell_texts[i:i+256]
            embeds = model.encode(batch, batch_size=256)
            all_query_embeds.append(embeds.cpu())
    query_embeds = torch.cat(all_query_embeds, dim=0)

    id_to_name = {str(e["id"]): e["name"] for e in kb_entities}
    preds, labels, pred_names, label_names, candidates_info = [], [], [], [], []

    for i in tqdm(range(0, len(samples), 256), desc="Searching"):
        q = query_embeds[i:i+256]
        scores = torch.matmul(q, kb_embeddings.t())
        topk_scores, topk_indices = torch.topk(scores, k=args.top_k, dim=-1)

        for j in range(q.size(0)):
            idx = i + j
            cands = [{"id": str(kb_entities[k]["id"]), "name": kb_entities[k]["name"]}
                     for k in topk_indices[j].tolist()]
            candidates_info.append(cands)
            preds.append(cands[0]["id"] if cands else "NIL")
            pred_names.append(cands[0]["name"] if cands else "NIL")
            labels.append(samples[idx]["gold_entity_id"])
            label_names.append(id_to_name.get(samples[idx]["gold_entity_id"],
                                               str(samples[idx]["gold_entity_id"])))

    metrics = evaluate_predictions(
        preds, labels,
        candidates_info=candidates_info,
        pred_names=pred_names,
        label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)


if __name__ == "__main__":
    main()
