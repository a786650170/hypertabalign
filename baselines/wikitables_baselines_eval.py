"""
Stand-alone wikitables_el cross-domain baseline eval.
Runs on 1043 GPU3. No dependence on baselines/eval_utils.py.

Baselines:
  vanilla    : raw DeBERTa-v3-base, CLS pool, zero-shot
  mpnet      : sentence-transformers all-mpnet-base-v2
  rsupcon    : a contrastively-trained DeBERTa-v3 biencoder ckpt (if path given)

For each baseline:
  1. Load model -> encode 3.97M wikitables KB names (cached to disk)
  2. Read gt.csv to get (cell_id, gold_wiki_title) labels (test split only)
  3. Extract cell text for each labeled cell from tables.json.gz
  4. Encode cell texts -> cosine top-K -> Acc/MRR/Hit
  5. Append a row to comparison_table.csv
"""
import argparse, json, os, sys, gzip, hashlib, time
import torch
import torch.nn.functional as F
from tqdm import tqdm


def hash_split(table_id: str) -> str:
    h = int(hashlib.md5(str(table_id).encode("utf-8")).hexdigest(), 16) % 10
    if h < 8: return "train"
    if h < 9: return "val"
    return "test"


# ----- Encoders -----
def encode_vanilla(texts, model_path, device, bs=256, max_len=64):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_path)
    m = AutoModel.from_pretrained(model_path).to(device).eval()
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in tqdm(range(0, len(texts), bs), desc="vanilla-enc"):
            b = texts[i:i+bs]
            inp = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            o = m(**inp).last_hidden_state[:, 0, :].float()
            out.append(F.normalize(o, p=2, dim=-1).cpu())
    return torch.cat(out, 0)


def encode_mpnet(texts, model_path, device, bs=256, max_len=128):
    """MPNet encoder using HF transformers (avoid sentence_transformers dep).
    Uses mean pooling over last_hidden_state masked by attention_mask, then L2 normalize."""
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_path)
    m = AutoModel.from_pretrained(model_path).to(device).eval()
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in tqdm(range(0, len(texts), bs), desc="mpnet-enc"):
            b = texts[i:i+bs]
            inp = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            o = m(**inp).last_hidden_state.float()
            mask = inp['attention_mask'].unsqueeze(-1).float()
            pooled = (o * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(F.normalize(pooled, p=2, dim=-1).cpu())
    return torch.cat(out, 0)


def encode_rsupcon(texts, model_path, ckpt_path, device, bs=256, max_len=64):
    """R-SupCon biencoder = DeBERTa + 2-layer proj head (768->768->256).
    Ckpt keys are prefixed with 'encoder.' (from BiEncoder wrapper); strip
    that prefix before loading into AutoModel. Then build proj head separately."""
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn
    tok = AutoTokenizer.from_pretrained(model_path)
    m = AutoModel.from_pretrained(model_path).to(device).eval()
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd_raw, dict) and "model" in sd_raw: sd_raw = sd_raw["model"]
    # split keys into encoder-side and proj-side
    enc_sd, proj_sd = {}, {}
    for k, v in sd_raw.items():
        k = k.removeprefix("module.")
        if k.startswith("encoder."):
            enc_sd[k[len("encoder."):]] = v
        elif k.startswith("proj."):
            proj_sd[k] = v
        # ignore temperature
    missing, unexpected = m.load_state_dict(enc_sd, strict=False)
    print(f"[rsupcon] encoder load: missing={len(missing)} unexpected={len(unexpected)} loaded={len(enc_sd)-len(unexpected)}/{len(enc_sd)}")
    # Build proj head Sequential(Linear(768,768), GELU/Tanh/ReLU, Linear(768,256))
    # GELU is most common in DeBERTa-aligned heads; pick GELU.
    proj = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 256)).to(device).eval()
    proj_strip = {k[len("proj."):]: v for k, v in proj_sd.items()}
    pmiss, punx = proj.load_state_dict(proj_strip, strict=False)
    print(f"[rsupcon] proj load: missing={pmiss} unexpected={punx}")
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in tqdm(range(0, len(texts), bs), desc="rsupcon-enc"):
            b = texts[i:i+bs]
            inp = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            cls = m(**inp).last_hidden_state[:, 0, :].float()  # CLS
            z = proj(cls)
            out.append(F.normalize(z, p=2, dim=-1).cpu())
    return torch.cat(out, 0)


# ----- Data loading -----
def load_labels(gt_path):
    """{table_id: { (row, col): gold_wiki_title }} for test split only."""
    out = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 4: continue
            cell_full = parts[1]
            ent_id = parts[3]
            try:
                tid, r_s, c_s = cell_full.rsplit("__", 2)
                r, c = int(r_s), int(c_s)
            except (ValueError, IndexError):
                continue
            if hash_split(tid) != "test": continue
            out.setdefault(tid, {})[(r, c)] = ent_id
    return out


def stream_tables(tables_path, wanted_ids):
    """Yield only wanted tables from tables.json.gz."""
    with gzip.open(tables_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            tid = d.get("_id")
            if tid in wanted_ids:
                yield d


def extract_cell_text(table_meta, r, c):
    """Return cell text at data row r, col c."""
    rows = table_meta.get("tableData") or []
    if r < 0 or r >= len(rows): return ""
    cells = rows[r]
    if c < 0 or c >= len(cells): return ""
    return (cells[c].get("text") or "").strip()


def build_row_context(table_meta, r):
    rows = table_meta.get("tableData") or []
    if r < 0 or r >= len(rows): return ""
    return " | ".join((c.get("text") or "").strip() for c in rows[r])


# ----- Main -----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, choices=["vanilla", "mpnet", "rsupcon"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--rsupcon_ckpt", default="")
    p.add_argument("--data_root", default="./data/wikitables_el")
    p.add_argument("--kb_cache", default="")  # auto-derived from baseline
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--tag", required=True)
    p.add_argument("--use_row_context", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Auto-derive cache path
    if not args.kb_cache:
        args.kb_cache = f"./experiments/kb_index/wikitables_baseline_{args.baseline}/kb_embeds.pt"
    os.makedirs(os.path.dirname(args.kb_cache), exist_ok=True)

    kb_path = os.path.join(args.data_root, "kb.jsonl")
    gt_path = os.path.join(args.data_root, "eval", "gt.csv")
    tables_path = os.path.join(args.data_root, "tables.json.gz")

    # 1. Load KB names
    print(f"[KB] reading {kb_path}")
    kb_ids, kb_names = [], []
    with open(kb_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            d = json.loads(line)
            kb_ids.append(d["id"]); kb_names.append(d.get("name", d["id"]))
    print(f"[KB] {len(kb_ids)} entities")
    id2name = dict(zip(kb_ids, kb_names))

    # 2. Encode (or load cache) KB
    if os.path.exists(args.kb_cache):
        print(f"[KB] loading cached embeds {args.kb_cache}")
        kb_emb = torch.load(args.kb_cache, map_location="cpu", weights_only=False).float()
    else:
        t0 = time.time()
        if args.baseline == "vanilla":
            kb_emb = encode_vanilla(kb_names, args.model_path, device, bs=args.bs)
        elif args.baseline == "mpnet":
            kb_emb = encode_mpnet(kb_names, args.model_path, device, bs=args.bs)
        elif args.baseline == "rsupcon":
            assert args.rsupcon_ckpt, "need --rsupcon_ckpt"
            kb_emb = encode_rsupcon(kb_names, args.model_path, args.rsupcon_ckpt, device, bs=args.bs)
        print(f"[KB] encoded {kb_emb.shape} in {time.time()-t0:.1f}s -> saving")
        torch.save(kb_emb, args.kb_cache)
    kb_emb_dev = F.normalize(kb_emb, p=2, dim=-1).to(device)

    # 3. Load labels
    print(f"[gt] reading {gt_path}")
    labels = load_labels(gt_path)
    print(f"[gt] tables-with-test-labels = {len(labels)}")
    wanted_ids = set(labels.keys())

    # 4. Extract eval cells from tables.json.gz
    print(f"[tbl] streaming tables.json.gz, extracting cells")
    eval_texts, eval_golds = [], []
    n_tables = 0
    for d in tqdm(stream_tables(tables_path, wanted_ids), total=len(wanted_ids)):
        tid = d["_id"]
        for (r, c), gold in labels[tid].items():
            txt = extract_cell_text(d, r, c)
            if args.use_row_context:
                ctx = build_row_context(d, r)
                txt = f"{txt} [SEP] {ctx}" if txt else ctx
            if not txt:
                txt = " "  # avoid empty
            eval_texts.append(txt); eval_golds.append(gold)
        n_tables += 1
    print(f"[eval] {len(eval_texts)} labeled cells from {n_tables} tables")

    # 5. Encode eval cells
    t0 = time.time()
    if args.baseline == "vanilla":
        q_emb = encode_vanilla(eval_texts, args.model_path, device, bs=args.bs)
    elif args.baseline == "mpnet":
        q_emb = encode_mpnet(eval_texts, args.model_path, device, bs=args.bs)
    elif args.baseline == "rsupcon":
        q_emb = encode_rsupcon(eval_texts, args.model_path, args.rsupcon_ckpt, device, bs=args.bs)
    q_emb = F.normalize(q_emb, p=2, dim=-1).to(device)
    print(f"[eval] queries encoded {q_emb.shape} in {time.time()-t0:.1f}s")

    # 6. Top-K retrieval, batched
    K = 10
    correct = 0; hits = {5: 0, 10: 0}; mrr10 = 0.0; name_correct = 0
    chunk = 256
    for i in tqdm(range(0, q_emb.size(0), chunk), desc="topk"):
        qc = q_emb[i:i+chunk]
        sc = torch.matmul(qc, kb_emb_dev.t())
        _, top = torch.topk(sc, k=K, dim=-1)
        top = top.cpu().tolist()
        for j in range(qc.size(0)):
            gold = eval_golds[i + j]
            top_ids = [kb_ids[k] for k in top[j]]
            if gold in top_ids:
                rank = top_ids.index(gold) + 1
                if rank <= 5: hits[5] += 1
                if rank <= 10: hits[10] += 1
                mrr10 += 1.0 / rank
                if rank == 1:
                    correct += 1
                    name_correct += 1  # name-acc == top-1 name match here

    n = q_emb.size(0)
    metrics = {
        "Accuracy": correct / n,
        "MRR@10": mrr10 / n,
        "Hit@5": hits[5] / n,
        "Hit@10": hits[10] / n,
        "Name-Accuracy": name_correct / n,
        "n": n,
    }
    print(f"[done] {args.tag}: {metrics}")

    # 7. Append to CSV
    csv = "./results/comparison_table.csv"
    line = f"\"{args.tag}\",{metrics['Accuracy']:.4f},{metrics['Accuracy']:.4f},{metrics['MRR@10']:.4f},{metrics['Hit@5']:.4f},{metrics['Hit@10']:.4f},{metrics['Name-Accuracy']:.4f},{metrics['Name-Accuracy']:.4f}\n"
    with open(csv, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[csv] appended -> {csv}")


if __name__ == "__main__":
    main()
