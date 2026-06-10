"""
True Zero-Shot Lower Bound: Vanilla pre-trained DeBERTa, NO fine-tuning at all.

Pipeline:
  1. Load vanilla DeBERTa-v3-base from HuggingFace (no checkpoint, no training).
  2. Encode each eval cell's text with CLS pooling + L2 normalize.
  3. Look up against the same 8M-entity KB index that other baselines use,
     reusing the cached vanilla DeBERTa KB embeddings (built once by
     dense_retriever.py — also from vanilla pre-trained weights, no checkpoint).
  4. Cosine top-K → Hits@K, MRR.

This is THE absolute floor: any method that beats this proves it learned
something, anything below this is worse than off-the-shelf BERT.
"""
import argparse
import os
import sys
import json

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from eval_utils import (
    load_eval_samples,
    evaluate_predictions,
    save_results,
    KB_PATH,
)


def cls_encode(model, tokenizer, texts, device, batch_size=128, max_length=128,
               use_fp16=True):
    """Pure CLS-pool encoding: no projection, no fine-tune, just AutoModel."""
    all_embeds = []
    model.eval()
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if use_fp16 and device.type == "cuda" else
        torch.cuda.amp.autocast(enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].float()
            all_embeds.append(F.normalize(cls, p=2, dim=-1).cpu())
    return torch.cat(all_embeds, dim=0)


def main():
    parser = argparse.ArgumentParser(
        description="Vanilla DeBERTa zero-shot retrieval baseline (no training)."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Local path or HF id (e.g. microsoft/deberta-v3-base)")
    parser.add_argument("--tag", type=str,
                        default="Vanilla DeBERTa (zero-shot, NO fine-tune)")
    parser.add_argument("--kb_path", type=str, default=KB_PATH)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=64,
                        help="Most KB entity names are short; 64 tokens is plenty.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable fp16 forward (default: enabled for 2x speedup).")
    parser.add_argument("--kb_cache", type=str,
                        default=os.path.join(PROJECT_ROOT,
                            "experiments/kb_index/vanilla_deberta_cls/kb_index.pt"),
                        help="Cache path for KB CLS embeddings (auto-built once).")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Vanilla baseline: {args.tag}")
    print(f"Model: {args.model_name}  (NO checkpoint, NO fine-tune)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("\n[1/4] Loading vanilla pre-trained encoder...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    print("\n[2/4] Loading KB...")
    kb_entities = []
    with open(args.kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_entities.append(json.loads(line))
    kb_names = [e["name"] for e in kb_entities]
    print(f"  KB size: {len(kb_entities)}")

    if os.path.exists(args.kb_cache):
        print(f"\n[3/4] Loading cached KB embeddings from {args.kb_cache} ...")
        kb_embeds = torch.load(args.kb_cache, map_location="cpu")
        kb_embeds = F.normalize(kb_embeds, p=2, dim=-1)
        assert kb_embeds.size(0) == len(kb_entities), \
            f"cache mismatch: {kb_embeds.size(0)} vs {len(kb_entities)}"
        print(f"  Loaded shape={tuple(kb_embeds.shape)}")
    else:
        print(f"\n[3/4] Building KB index (one-time, vanilla CLS pooling)...")
        kb_embeds = cls_encode(model, tokenizer, kb_names, device,
                               batch_size=args.batch_size,
                               max_length=args.max_length,
                               use_fp16=not args.no_fp16)
        os.makedirs(os.path.dirname(args.kb_cache), exist_ok=True)
        torch.save(kb_embeds, args.kb_cache)
        print(f"  Saved cache to {args.kb_cache}")

    print("\n[4/4] Encoding eval queries + retrieval...")
    eval_samples = load_eval_samples()
    cell_texts = [s["cell_text"] for s in eval_samples]
    gold_ids = [s["gold_entity_id"] for s in eval_samples]
    query_embeds = cls_encode(model, tokenizer, cell_texts, device,
                              batch_size=args.batch_size, max_length=128,
                              use_fp16=not args.no_fp16)

    kb_embeds_dev = kb_embeds.to(device)
    preds, pred_names = [], []
    candidates_info, label_names = [], []
    chunk = 256
    with torch.no_grad():
        for i in range(0, query_embeds.size(0), chunk):
            q = query_embeds[i:i + chunk].to(device)
            scores = torch.matmul(q, kb_embeds_dev.t())
            topk_scores, topk_idx = torch.topk(scores, k=args.top_k, dim=-1)
            for j in range(q.size(0)):
                cands = []
                for k in range(args.top_k):
                    eidx = topk_idx[j, k].item()
                    cands.append({
                        "id": str(kb_entities[eidx]["id"]),
                        "name": kb_entities[eidx]["name"],
                        "score": float(topk_scores[j, k].item()),
                    })
                preds.append(cands[0]["id"])
                pred_names.append(cands[0]["name"])
                candidates_info.append(cands)
                label_names.append("")

    metrics = evaluate_predictions(
        preds, gold_ids,
        candidates_info=candidates_info,
        pred_names=pred_names,
        label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)
    print(f"\n[done] {args.tag}")


if __name__ == "__main__":
    main()
