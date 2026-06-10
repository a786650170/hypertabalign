"""MPNet zero-shot retrieval on WDC LSPM, with wall-time timing.

KB index build + eval-set encoding + cosine top-k, all on a single GPU.
"""
import os, sys, json, time, argparse, csv
PROJECT_ROOT = "."
sys.path.insert(0, PROJECT_ROOT)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from tqdm import tqdm


def encode_with_st(texts, model, device, bs=256):
    """Encode with sentence-transformers; returns L2-normalised CPU tensor."""
    embs = []
    for i in tqdm(range(0, len(texts), bs), desc="enc"):
        batch = texts[i:i + bs]
        out = model.encode(batch, convert_to_tensor=True, device=device,
                           normalize_embeddings=True, show_progress_bar=False)
        embs.append(out.detach().cpu())
    return torch.cat(embs, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",
                   default="./models_cache/_modelscope_cache/sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--kb_path",
                   default="./data/wdc_lspm_sampled/wdc_products_kb.jsonl")
    p.add_argument("--gt_path",
                   default="./data/wdc_lspm_sampled/eval/gt.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--tag", default="MPNet WDC zero-shot latency")
    p.add_argument("--out_csv", default="./results/comparison_table.csv")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[load] {args.model_path}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_path, device=str(device))
    model.eval()

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
    kb_emb = encode_with_st(kb_names, model, device, bs=args.bs)  # [M, D]
    print(f"[KB] embed shape {tuple(kb_emb.shape)} dtype {kb_emb.dtype}")
    t_kb_end = time.time()
    print(f"[KB] index build wall time: {(t_kb_end - t_kb_start)/60:.2f} min")

    # ---------- 3) read eval queries ----------
    print(f"[Eval] reading {args.gt_path}")
    eval_texts, gold_ids = [], []
    with open(args.gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ct = (row.get("cell_text") or row.get("text") or row.get("cell") or "").strip()
            gid = (row.get("entity_id") or row.get("gold_id") or row.get("id") or "").strip()
            if ct:
                eval_texts.append(ct)
                gold_ids.append(gid)
    print(f"[Eval] {len(eval_texts)} queries")

    # ---------- 4) encode queries ----------
    t_q_start = time.time()
    q_emb = encode_with_st(eval_texts, model, device, bs=args.bs)
    t_q_end = time.time()
    print(f"[Eval] query encode wall time: {(t_q_end - t_q_start)/60:.2f} min")

    # ---------- 5) chunked top-k cosine ----------
    print(f"[Search] chunked top-k cosine, chunk=128")
    kb_emb_dev = kb_emb.to(device)  # [M, D]
    chunk = 128
    all_top_ids = []
    t_s_start = time.time()
    for i in tqdm(range(0, q_emb.size(0), chunk), desc="search"):
        q = q_emb[i:i + chunk].to(device)
        sim = q @ kb_emb_dev.t()  # [chunk, M]
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

    # append to comparison csv
    try:
        with open(args.out_csv, "a", encoding="utf-8") as f:
            f.write(f"\"{args.tag}\",{metrics['Accuracy']:.4f},{metrics['Accuracy']:.4f},"
                    f"{metrics['MRR_at_10']:.4f},{metrics['Hit_at_5']:.4f},{metrics['Hit_at_10']:.4f},"
                    f"-, -, total_min={t_total/60:.2f}, ms_per_q={t_total*1000/max(n,1):.2f}\n")
    except Exception as e:
        print(f"[csv append] {e}")


if __name__ == "__main__":
    main()
