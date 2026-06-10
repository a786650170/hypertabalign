"""
Vanilla BERT Bi-Encoder Baseline (Lower Bound).
Uses pre-trained bert-base-uncased WITHOUT any fine-tuning.
Encodes KB entity names and query cell texts with [CLS] pooling,
retrieves by cosine similarity.

This represents the lower bound: a general-purpose pre-trained encoder
with no domain training, no table structure, no contrastive learning.
"""
import argparse
import os
import sys
import json

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_utils import load_eval_samples, load_kb, evaluate_predictions, save_results, KB_PATH


class BERTEncoder:
    """Vanilla BERT encoder with [CLS] pooling, no fine-tuning."""

    def __init__(self, model_name="bert-base-uncased", device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[BERT] Loaded {model_name} (frozen, no fine-tuning)")

    @torch.no_grad()
    def encode(self, texts, batch_size=256):
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            ).to(self.device)
            outputs = self.model(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :]
            cls_embeds = F.normalize(cls_embeds, p=2, dim=-1)
            all_embeds.append(cls_embeds.cpu())
        return torch.cat(all_embeds, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Vanilla BERT Bi-Encoder Baseline")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--kb_path", type=str, default=KB_PATH)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("Vanilla BERT Bi-Encoder Baseline (Lower Bound)")
    print(f"Model: {args.model_name} (NO fine-tuning)")
    print("=" * 60)

    encoder = BERTEncoder(model_name=args.model_name, device=args.device)

    samples = load_eval_samples()
    kb = load_kb(kb_path=args.kb_path)

    cache_dir = os.path.join(PROJECT_ROOT, "experiments/kb_index/bert_baseline")
    index_path = os.path.join(cache_dir, "kb_index.pt")
    meta_path = os.path.join(cache_dir, "kb_meta.jsonl")

    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("\n[1/3] Loading cached BERT KB index...")
        kb_embeddings = torch.load(index_path, map_location="cpu")
        kb_embeddings = F.normalize(kb_embeddings, p=2, dim=-1)
        kb_entities = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                kb_entities.append(json.loads(line))
        print(f"  Loaded {len(kb_entities)} entity embeddings, shape={kb_embeddings.shape}")
    else:
        print(f"\n[1/3] Encoding {len(kb)} KB entities with BERT (first time, will cache)...")
        os.makedirs(cache_dir, exist_ok=True)
        kb_names = [e["name"] for e in kb]
        kb_embeddings = encoder.encode(kb_names, batch_size=args.encode_batch_size)
        torch.save(kb_embeddings, index_path)
        kb_entities = kb
        with open(meta_path, "w", encoding="utf-8") as f:
            for e in kb:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  Saved index: {kb_embeddings.shape}")

    print(f"\n[2/3] Encoding {len(samples)} eval queries...")
    cell_texts = [s["cell_text"] for s in samples]
    query_embeds = encoder.encode(cell_texts, batch_size=args.encode_batch_size)

    print(f"\n[3/3] Retrieving top-{args.top_k}...")
    id_to_name = {str(e["id"]): e["name"] for e in kb_entities}
    preds, labels, pred_names, label_names, candidates_info = [], [], [], [], []

    chunk_size = 512
    for i in tqdm(range(0, len(samples), chunk_size), desc="Searching"):
        q = query_embeds[i:i+chunk_size]
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
        tag="BERT (vanilla)",
    )
    save_results(metrics, "BERT (vanilla)")


if __name__ == "__main__":
    main()
