"""Per-cell top-1 dump for zero-shot text bi-encoders (MPNet / Vanilla DeBERTa).

Output JSONL line per labeled eval cell:
  table_idx, cell_idx, cell_text, gold_id, gold_name,
  top1_id, top1_name, top1_score, top5_ids, top5_names

Use --model {mpnet, vanilla}.
"""
import os, sys, json, time, argparse
sys.path.insert(0, ".")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import transformers.utils.import_utils as _iu
    if hasattr(_iu, 'check_torch_load_is_safe'):
        _iu.check_torch_load_is_safe = lambda: True
    import transformers.modeling_utils as _mu
    if hasattr(_mu, 'check_torch_load_is_safe'):
        _mu.check_torch_load_is_safe = lambda: True
except Exception:
    pass

from data.dataset import TableAlignmentDataset


def encode_mpnet(model, texts, dev, bs=256, desc="enc"):
    embs = []
    for i in tqdm(range(0, len(texts), bs), desc=desc, mininterval=15.0):
        batch = texts[i:i + bs]
        out = model.encode(batch, convert_to_tensor=True, device=str(dev),
                           normalize_embeddings=True, show_progress_bar=False)
        embs.append(out.detach().half().to('cpu'))
    return torch.cat(embs, dim=0)


def encode_vanilla(model, tok, texts, dev, bs=256, max_len=128, desc="enc"):
    embs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc=desc, mininterval=15.0):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(dev)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :]
            cls = F.normalize(cls, p=2, dim=-1).half()
            embs.append(cls.detach().to('cpu'))
    return torch.cat(embs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['mpnet', 'vanilla'])
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--eval_dir', default='./data/wdc_lspm_sampled/eval')
    ap.add_argument('--kb_path', default='./data/wdc_lspm_sampled/wdc_products_kb.jsonl')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--max_tables', type=int, default=8000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)

    print(f"[load] {args.model} from {args.model_path}", flush=True)
    if args.model == 'mpnet':
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model_path, device=str(dev))
        model.eval()
        encoder = lambda texts, desc: encode_mpnet(model, texts, dev, bs=args.bs, desc=desc)
    else:  # vanilla
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path).to(dev)
        encoder = lambda texts, desc: encode_vanilla(model, tok, texts, dev, bs=args.bs, desc=desc)

    # ---------- KB encode ----------
    print(f"[KB] reading {args.kb_path}", flush=True)
    kb_ids, kb_names = [], []
    with open(args.kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            kb_ids.append(d['id']); kb_names.append(d.get('name', d['id']))
    M = len(kb_ids)
    kb_id_to_idx = {k: i for i, k in enumerate(kb_ids)}
    print(f"[KB] {M} entities", flush=True)

    t0 = time.time()
    kb_emb = encoder(kb_names, "kb_enc").to(dev)
    print(f"[KB] shape={tuple(kb_emb.shape)} time={(time.time()-t0)/60:.2f}min", flush=True)

    # ---------- collect labeled cells ----------
    print(f"[Eval] loading {args.eval_dir}", flush=True)
    ds = TableAlignmentDataset(root=args.eval_dir)
    n_lim = min(args.max_tables, len(ds))
    print(f"[Eval] tables={len(ds)}, dumping up to {n_lim}", flush=True)

    queries = []
    for ti in tqdm(range(n_lim), desc="extract", mininterval=15.0):
        try: g = ds[ti]
        except Exception: continue
        if getattr(g, 'is_dummy', False): continue
        labeled = g.labeled_indices.tolist() if hasattr(g.labeled_indices, 'tolist') else list(g.labeled_indices)
        gold_lst = list(getattr(g, 'target_ent_ids', []))
        for idx, gid in zip(labeled, gold_lst):
            if 0 <= idx < len(g.x_text):
                txt = str(g.x_text[idx]).strip()
                gid_str = (gid[0] if isinstance(gid, (list, tuple)) and gid else gid) or ''
                gid_str = str(gid_str).strip()
                if txt and gid_str:
                    queries.append((ti, int(idx), txt, gid_str))
    print(f"[Q] {len(queries)} labeled cells", flush=True)

    cell_texts = [q[2] for q in queries]
    t1 = time.time()
    q_emb = encoder(cell_texts, "q_enc")
    print(f"[Q] shape={tuple(q_emb.shape)} time={(time.time()-t1)/60:.2f}min", flush=True)

    # ---------- search ----------
    out = open(args.out, 'w', encoding='utf-8')
    chunk = 128
    for i in tqdm(range(0, q_emb.size(0), chunk), desc="search", mininterval=10.0):
        q = q_emb[i:i + chunk].to(dev).float()
        sim = q @ kb_emb.float().t()
        top_vals, top_inds = torch.topk(sim, args.topk, dim=-1)
        top_inds = top_inds.detach().cpu().tolist()
        top_vals = top_vals.detach().cpu().tolist()
        for r, (ti, ci, txt, gid) in enumerate(queries[i:i + chunk]):
            gold_idx = kb_id_to_idx.get(gid)
            gold_name = kb_names[gold_idx] if gold_idx is not None else None
            t1_kb = top_inds[r][0]
            rec = dict(
                table_idx=ti, cell_idx=ci, cell_text=txt,
                gold_id=gid, gold_name=gold_name,
                top1_id=kb_ids[t1_kb], top1_name=kb_names[t1_kb], top1_score=float(top_vals[r][0]),
                top5_ids=[kb_ids[j] for j in top_inds[r]],
                top5_names=[kb_names[j] for j in top_inds[r]],
            )
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    out.close()
    print(f"[done] dumped {len(queries)} cells -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
