"""
DeBERTa Bi-Encoder Baseline (No Fine-tuning).
Uses pre-trained DeBERTa-v3-base WITHOUT any task-specific training.
Encodes cell text and KB entity names with KBEncoder, retrieves by cosine similarity.

This represents the pre-trained dense retrieval baseline:
same encoder architecture as ours, but no training, no GNN, no structure.
"""
import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"
import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from eval_utils import load_eval_samples, evaluate_predictions, save_results, KB_PATH
from models.encoder.text_backbone import TextEncoder
from models.knowledge.kb_encoder import KBEncoder


def main():
    parser = argparse.ArgumentParser(description="DeBERTa Bi-Encoder Baseline (vanilla)")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--kb_path", type=str, default=KB_PATH)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("DeBERTa Bi-Encoder Baseline (Pre-trained, NO fine-tuning)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    text_encoder = TextEncoder(args.model_name, multi_gpu=False)
    kb_encoder = KBEncoder(text_encoder=text_encoder)
    text_encoder.to(device)
    kb_encoder.to(device)
    kb_encoder.eval()
    print(f"[DeBERTa] Using vanilla pre-trained {args.model_name} (NO checkpoint loaded)")

    samples = load_eval_samples()

    cache_dir = os.path.join(PROJECT_ROOT, "experiments/kb_index/biencoder_vanilla")
    kb_index_path = os.path.join(cache_dir, "kb_index.pt")
    kb_meta_path = os.path.join(cache_dir, "kb_meta.jsonl")

    if os.path.exists(kb_index_path) and os.path.exists(kb_meta_path):
        print("\n[1/3] Loading cached vanilla DeBERTa KB index...")
        kb_embeddings = torch.load(kb_index_path, map_location="cpu")
        kb_embeddings = F.normalize(kb_embeddings, p=2, dim=-1)
        kb_entities = []
        with open(kb_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                kb_entities.append(json.loads(line))
        print(f"  Loaded {len(kb_entities)} entity embeddings")
    else:
        print(f"\n[1/3] Encoding KB entities with vanilla DeBERTa (first time, will cache)...")
        os.makedirs(cache_dir, exist_ok=True)
        kb_entities = []
        with open(args.kb_path, "r", encoding="utf-8") as f:
            for line in f:
                kb_entities.append(json.loads(line))

        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(kb_entities), 1024), desc="Encoding KB"):
                batch = kb_entities[i:i+1024]
                names = [e["name"] for e in batch]
                paths = [e.get("path", "") for e in batch]
                embeds = kb_encoder(names, paths, text_micro_batch_size=64, path_chunk_size=512)
                all_embeds.append(embeds.cpu())

        kb_embeddings = torch.cat(all_embeds, dim=0)
        kb_embeddings = F.normalize(kb_embeddings, p=2, dim=-1)
        torch.save(kb_embeddings, kb_index_path)
        with open(kb_meta_path, "w", encoding="utf-8") as f:
            for e in kb_entities:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  Saved index: {kb_embeddings.shape}")

    print(f"\n[2/3] Encoding {len(samples)} eval cell texts...")
    cell_texts = [s["cell_text"] for s in samples]

    all_query_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(cell_texts), args.batch_size), desc="Encoding queries"):
            batch_texts = cell_texts[i:i+args.batch_size]
            embeds = kb_encoder(batch_texts, paths=[""] * len(batch_texts),
                                text_micro_batch_size=args.batch_size)
            all_query_embeds.append(embeds.cpu())

    query_embeds = torch.cat(all_query_embeds, dim=0)
    query_embeds = F.normalize(query_embeds, p=2, dim=-1)

    print(f"\n[3/3] Retrieving top-{args.top_k}...")
    preds, labels, pred_names, label_names, candidates_info = [], [], [], [], []
    id_to_name = {str(e["id"]): e["name"] for e in kb_entities}

    chunk_size = 256
    for i in tqdm(range(0, len(samples), chunk_size), desc="Searching"):
        q_chunk = query_embeds[i:i+chunk_size]
        scores = torch.matmul(q_chunk, kb_embeddings.t())
        topk_scores, topk_indices = torch.topk(scores, k=args.top_k, dim=-1)

        for j in range(q_chunk.size(0)):
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
        tag="DeBERTa Bi-Encoder (vanilla)",
    )
    save_results(metrics, "DeBERTa Bi-Encoder (vanilla)")


if __name__ == "__main__":
    main()
