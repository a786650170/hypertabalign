"""
Build vanilla-DeBERTa-CLS top-K candidates for all eval queries and persist
them to disk, so that downstream LLM rerank baselines can run in a different
conda env (with newer transformers) without re-doing the broken-in-newer-TF
encoding step.

Run this script under env tyf1 (transformers==5.0.0, torch>=2.5) -- that is
the configuration in which the cached KB index was originally built.
"""
import os, sys, json, argparse, pickle
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm

import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from eval_utils import load_eval_samples, KB_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deberta_path",
                   default="./models_cache/deberta-v3-base")
    p.add_argument("--kb_cache",
                   default="./experiments/kb_index/vanilla_deberta_cls/kb_index.pt")
    p.add_argument("--kb_path", default=KB_PATH)
    p.add_argument("--out_pkl",
                   default="./experiments/kb_index/vanilla_deberta_cls/candidates_top50.pkl")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--enc_bs", type=int, default=256)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    eval_samples = load_eval_samples()
    if args.limit > 0:
        eval_samples = eval_samples[:args.limit]
    print(f"[eval] {len(eval_samples)} samples")

    print(f"[retrieve] loading DeBERTa from {args.deberta_path}")
    tok = AutoTokenizer.from_pretrained(args.deberta_path)
    model = AutoModel.from_pretrained(args.deberta_path).to(device).eval()

    print(f"[retrieve] loading KB jsonl {args.kb_path}")
    kb_entities = []
    with open(args.kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_entities.append(json.loads(line))
    print(f"[retrieve] KB size = {len(kb_entities)}")

    print(f"[retrieve] loading cached KB CLS embeddings {args.kb_cache}")
    kb_embeds = torch.load(args.kb_cache, map_location="cpu", weights_only=False).float()
    kb_embeds = F.normalize(kb_embeds, p=2, dim=-1).to(device)
    assert kb_embeds.size(0) == len(kb_entities)

    cells = [s["cell_text"] for s in eval_samples]
    q_all = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in tqdm(range(0, len(cells), args.enc_bs), desc="encode-q"):
            batch = cells[i:i + args.enc_bs]
            inp = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len).to(device)
            out = model(**inp)
            cls = out.last_hidden_state[:, 0, :].float()
            q_all.append(F.normalize(cls, p=2, dim=-1).cpu())
    q_embeds = torch.cat(q_all, 0).to(device)
    print(f"[retrieve] query embeddings = {tuple(q_embeds.shape)}")

    candidates_per_q = []
    chunk = 128
    for i in tqdm(range(0, q_embeds.size(0), chunk), desc="topk"):
        q = q_embeds[i:i + chunk]
        scores = torch.matmul(q, kb_embeds.t())
        _, idx = torch.topk(scores, k=args.top_k, dim=-1)
        idx_cpu = idx.cpu().tolist()
        for j in range(q.size(0)):
            cands = [{"id": str(kb_entities[k]["id"]),
                      "name": kb_entities[k]["name"]} for k in idx_cpu[j]]
            candidates_per_q.append(cands)

    # quick Hit@K sanity check
    golds = [str(s["gold_entity_id"]) for s in eval_samples]
    hit5 = sum(1 for i, g in enumerate(golds) if g in [c["id"] for c in candidates_per_q[i][:5]])
    hit10 = sum(1 for i, g in enumerate(golds) if g in [c["id"] for c in candidates_per_q[i][:10]])
    print(f"[retrieve] sanity Hit@5={hit5}/{len(golds)} = {hit5/len(golds):.4f}")
    print(f"[retrieve] sanity Hit@10={hit10}/{len(golds)} = {hit10/len(golds):.4f}")

    os.makedirs(os.path.dirname(args.out_pkl), exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump({
            "candidates_per_q": candidates_per_q,
            "gold_entity_ids": golds,
            "kb_id_to_name": {str(e["id"]): e["name"] for e in kb_entities},
        }, f, protocol=4)
    print(f"[retrieve] wrote {args.out_pkl}")


if __name__ == "__main__":
    main()
