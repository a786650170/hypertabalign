"""
Unified text-retrieval baseline runner for sentence-transformer-style and
Qwen3-Embedding-style models.

Two model families are supported via `--model_type`:
  * `st`        : standard SentenceTransformer (MPNet, BGE, etc.). Uses the
                  library's encode() so each model's correct pooling/normalisation
                  is applied automatically.
  * `qwen3emb`  : Qwen3-Embedding-* (0.6B / 4B / 8B). Uses last-token pooling
                  per Qwen team's reference inference recipe, with the model's
                  recommended query/passage instructions.

Common pipeline (same as the existing Vanilla DeBERTa baseline):
  1. encode the 8M-entity KB once -> normalised float32 tensor, cached
  2. encode the 114k eval cells
  3. cosine top-K (K=top_k, default 10) on a single H100, chunked
  4. evaluate via the project's standard Evaluator (Acc / MRR@10 / Hit@5/10 / Name-Acc)
"""
import argparse
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from eval_utils import load_eval_samples, evaluate_predictions, save_results, KB_PATH


# ---------------------------------------------------------------------------
# Encoder backends
# ---------------------------------------------------------------------------
def make_st_encoder(model_path, device):
    """SentenceTransformer family: MPNet, BGE, etc."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    model.eval()

    @torch.no_grad()
    def encode(texts, batch_size=128, prefix=""):
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        embeds = model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=False, convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embeds.to(torch.float32)
    return encode


def make_qwen3emb_encoder(model_path, device):
    """Qwen3-Embedding-*: last-token pool, instruction-aware."""
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    max_len = 512

    @torch.no_grad()
    def encode(texts, batch_size=32, prefix=""):
        if prefix:
            texts = [f"{prefix}{t}" for t in texts]
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            h = model(**enc).last_hidden_state          # (B, T, D)
            # Left padding => last real token is index -1 of attention mask.
            # With left padding, the EOS lives at position -1 unconditionally.
            pooled = h[:, -1, :].to(torch.float32)
            pooled = F.normalize(pooled, p=2, dim=-1)
            out.append(pooled.cpu())
        return torch.cat(out, dim=0)
    return encode


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=["st", "qwen3emb"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--kb_cache_dir", required=True,
                   help="Per-model cache dir for KB embeddings.")
    p.add_argument("--query_prefix", default="",
                   help="Optional prefix for queries (e.g. 'query: ' for BGE).")
    p.add_argument("--passage_prefix", default="",
                   help="Optional prefix for KB entity names (e.g. 'passage: ').")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kb_path", default=KB_PATH)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[{args.tag}] device={device} model={args.model_path} type={args.model_type}")

    encode = (make_st_encoder if args.model_type == "st" else make_qwen3emb_encoder)(
        args.model_path, device
    )

    # --- Eval samples ---
    samples = load_eval_samples()
    cell_texts = [s["cell_text"] for s in samples]
    labels = [s["gold_entity_id"] for s in samples]
    print(f"[{args.tag}] eval samples = {len(samples)}")

    # --- KB index ---
    os.makedirs(args.kb_cache_dir, exist_ok=True)
    kb_index_path = os.path.join(args.kb_cache_dir, "kb_index.pt")
    kb_meta_path = os.path.join(args.kb_cache_dir, "kb_meta.jsonl")

    if os.path.exists(kb_index_path) and os.path.exists(kb_meta_path):
        print(f"[{args.tag}] loading cached KB index ...")
        kb_embeddings = torch.load(kb_index_path, map_location="cpu")
        kb_embeddings = F.normalize(kb_embeddings, p=2, dim=-1)
        kb_entities = []
        with open(kb_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                kb_entities.append(json.loads(line))
    else:
        print(f"[{args.tag}] encoding KB (one-time) ...")
        kb_entities = []
        with open(args.kb_path, "r", encoding="utf-8") as f:
            for line in f:
                kb_entities.append(json.loads(line))
        names = [e["name"] for e in kb_entities]

        kb_batches = []
        bs = args.batch_size
        for i in tqdm(range(0, len(names), bs), desc="Encoding KB"):
            kb_batches.append(encode(names[i:i + bs],
                                     batch_size=bs,
                                     prefix=args.passage_prefix).cpu())
        kb_embeddings = torch.cat(kb_batches, 0).to(torch.float32)
        kb_embeddings = F.normalize(kb_embeddings, p=2, dim=-1)
        torch.save(kb_embeddings, kb_index_path)
        with open(kb_meta_path, "w", encoding="utf-8") as f:
            for e in kb_entities:
                f.write(json.dumps({"id": e["id"], "name": e["name"]}) + "\n")
    print(f"[{args.tag}] KB shape = {tuple(kb_embeddings.shape)}")
    kb_embeddings_gpu = kb_embeddings.to(device)

    # --- Query encoding ---
    print(f"[{args.tag}] encoding {len(cell_texts)} queries ...")
    q_batches = []
    bs = args.batch_size
    for i in tqdm(range(0, len(cell_texts), bs), desc="Encoding queries"):
        q_batches.append(encode(cell_texts[i:i + bs],
                                batch_size=bs,
                                prefix=args.query_prefix).cpu())
    q_embeds = torch.cat(q_batches, 0).to(torch.float32)
    q_embeds = F.normalize(q_embeds, p=2, dim=-1)

    # --- Retrieval ---
    print(f"[{args.tag}] top-{args.top_k} retrieval ...")
    preds, pred_names, candidates_info, label_names = [], [], [], []
    id_to_name = {str(e["id"]): e["name"] for e in kb_entities}
    chunk = 256
    for i in tqdm(range(0, q_embeds.size(0), chunk), desc="Searching"):
        q = q_embeds[i:i + chunk].to(device)
        scores = torch.matmul(q, kb_embeddings_gpu.t())
        topk_scores, topk_idx = torch.topk(scores, k=args.top_k, dim=-1)
        topk_idx_cpu = topk_idx.cpu().tolist()
        for j in range(q.size(0)):
            cands = [{"id": str(kb_entities[k]["id"]),
                      "name": kb_entities[k]["name"]}
                     for k in topk_idx_cpu[j]]
            candidates_info.append(cands)
            preds.append(cands[0]["id"])
            pred_names.append(cands[0]["name"])
            idx = i + j
            label_names.append(id_to_name.get(labels[idx], str(labels[idx])))

    # --- Eval ---
    metrics = evaluate_predictions(
        preds, labels,
        candidates_info=candidates_info,
        pred_names=pred_names, label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)
    print(f"[done] {args.tag}")


if __name__ == "__main__":
    main()
