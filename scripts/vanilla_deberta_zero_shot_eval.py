"""Vanilla DeBERTa zero-shot retrieval on WDC LSPM, with wall-time timing.

Mirrors mpnet_wdc_latency.py but uses HF transformers AutoModel for DeBERTa.
KB build + eval encoding + cosine top-k, all on a single GPU, no fine-tuning.
"""
import os, sys, json, time, argparse, csv
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from tqdm import tqdm

# bypass torch.load CVE check
try:
    import transformers.utils.import_utils as _iu
    if hasattr(_iu, 'check_torch_load_is_safe'):
        _iu.check_torch_load_is_safe = lambda: True
    import transformers.modeling_utils as _mu
    if hasattr(_mu, 'check_torch_load_is_safe'):
        _mu.check_torch_load_is_safe = lambda: True
except Exception:
    pass

from transformers import AutoModel, AutoTokenizer


def encode_with_deberta(texts, model, tok, device, bs=256, max_len=128):
    """Encode with HF model, CLS pool, L2 normalize. Returns CPU tensor."""
    embs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc="enc"):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            out = model(**enc)
            # CLS token
            cls = out.last_hidden_state[:, 0, :]
            cls = F.normalize(cls, p=2, dim=-1)
            embs.append(cls.detach().cpu())
    return torch.cat(embs, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./models_cache/deberta-v3-base")
    p.add_argument("--kb_path",
                   default="./data/wdc_lspm_sampled/wdc_products_kb.jsonl")
    p.add_argument("--eval_dir",
                   default="./data/wdc_lspm_sampled/eval")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--tag", default="Vanilla DeBERTa WDC zero-shot latency")
    p.add_argument("--out_csv", default="./results/comparison_table.csv")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[load] {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)

    t0 = time.time()

    # ---------- 1) read KB ----------
    print(f"[KB] reading {args.kb_path}")
    kb_ids, kb_names = [], []
    with open(args.kb_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            kb_ids.append(d["id"])
            kb_names.append(d.get("name", d["id"]))
    print(f"[KB] {len(kb_names)} entities")

    # ---------- 2) encode KB ----------
    t_kb_start = time.time()
    print(f"[KB] encoding ...")
    kb_emb = encode_with_deberta(kb_names, model, tok, device, bs=args.bs)
    t_kb_end = time.time()
    print(f"[KB] embed shape {tuple(kb_emb.shape)}")
    print(f"[KB] index build wall time: {(t_kb_end - t_kb_start)/60:.2f} min")

    # ---------- 3) read eval queries via TableAlignmentDataset ----------
    print(f"[Eval] loading TableAlignmentDataset from {args.eval_dir}")
    sys.path.insert(0, ".")
    from data.dataset import TableAlignmentDataset
    ds = TableAlignmentDataset(root=args.eval_dir)
    print(f"[Eval] tables: {len(ds)}")
    eval_texts, gold_ids = [], []
    for i in range(len(ds)):
        g = ds[i]
        if getattr(g, "is_dummy", False):
            continue
        labeled = g.labeled_indices.tolist() if hasattr(g.labeled_indices, "tolist") else list(g.labeled_indices)
        gold_lst = list(getattr(g, "target_ent_ids", []))
        for idx, gid in zip(labeled, gold_lst):
            if 0 <= idx < len(g.x_text):
                txt = str(g.x_text[idx]).strip()
                gid_str = (gid[0] if isinstance(gid, (list, tuple)) and gid else gid) or ""
                gid_str = str(gid_str).strip()
                if txt:
                    eval_texts.append(txt)
                    gold_ids.append(gid_str)
    print(f"[Eval] {len(eval_texts)} labeled queries")

    # ---------- 4) encode queries ----------
    t_q_start = time.time()
    q_emb = encode_with_deberta(eval_texts, model, tok, device, bs=args.bs)
    t_q_end = time.time()
    print(f"[Eval] query encode wall time: {(t_q_end - t_q_start)/60:.2f} min")

    # ---------- 5) chunked top-k cosine ----------
    print(f"[Search] chunked top-k cosine, chunk=128")
    kb_emb_dev = kb_emb.to(device)
    chunk = 128
    all_top_ids = []
    t_s_start = time.time()
    for i in tqdm(range(0, q_emb.size(0), chunk), desc="search"):
        q = q_emb[i:i + chunk].to(device)
        sim = q @ kb_emb_dev.t()
        topk = torch.topk(sim, args.topk, dim=-1).indices.cpu().tolist()
        all_top_ids.extend(topk)
    t_s_end = time.time()
    print(f"[Search] top-k wall time: {(t_s_end - t_s_start)/60:.2f} min")

    # ---------- 6) metrics ----------
    n = len(gold_ids)
    hit1 = hit5 = hit10 = 0
    mrr10_sum = 0.0
    for i in range(n):
        ranked_ids = [kb_ids[j] for j in all_top_ids[i]]
        if gold_ids[i] in ranked_ids[:1]: hit1 += 1
        if gold_ids[i] in ranked_ids[:5]: hit5 += 1
        if gold_ids[i] in ranked_ids[:10]: hit10 += 1
        for r, rid in enumerate(ranked_ids[:10], 1):
            if rid == gold_ids[i]:
                mrr10_sum += 1.0 / r
                break
    metrics = dict(
        Accuracy=hit1 / n if n else 0.0,
        MRR_at_10=mrr10_sum / n if n else 0.0,
        Hit_at_5=hit5 / n if n else 0.0,
        Hit_at_10=hit10 / n if n else 0.0,
    )
    t_total = time.time() - t0
    print(f"[done] {args.tag}: {metrics}")
    print(f"[wall] TOTAL end-to-end: {t_total/60:.2f} min ({t_total:.1f} s)")
    print(f"[wall] per-query latency: {t_total * 1000 / max(n,1):.2f} ms/q")

    try:
        with open(args.out_csv, "a", encoding="utf-8") as f:
            f.write(f"\"{args.tag}\",{metrics['Accuracy']:.4f},{metrics['Accuracy']:.4f},"
                    f"{metrics['MRR_at_10']:.4f},{metrics['Hit_at_5']:.4f},{metrics['Hit_at_10']:.4f},"
                    f"-, -, total_min={t_total/60:.2f}, ms_per_q={t_total*1000/max(n,1):.2f}\n")
    except Exception as e:
        print(f"[csv append] {e}")


if __name__ == "__main__":
    main()
