"""R-SupCon per-cell top-1 dump on WDC LSPM 8M (companion to HyperTab dump).

Output JSONL line per labeled eval cell:
  table_idx, cell_idx, cell_text, gold_id, gold_name,
  top1_id, top1_name, top1_score, top5_ids, top5_names
"""
import os, sys, json, time, argparse
sys.path.insert(0, ".")
sys.path.insert(0, "./baselines")
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
from baselines.rsupcon_baseline import RSupConModel


def _mean_pool(out, attn):
    expanded = attn.unsqueeze(-1).expand(out.size()).float()
    return (out * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)


def encode(model, tok, texts, dev, bs=128, max_len=128, desc="enc"):
    embs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc=desc, mininterval=15.0):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(dev)
            out = model.encoder(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state
            pooled = _mean_pool(out, enc['attention_mask'])
            pooled = F.normalize(pooled, p=2, dim=-1).half()
            embs.append(pooled.detach().to('cpu'))
    return torch.cat(embs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='R-SupCon checkpoint path on this host')
    ap.add_argument('--model_name', default='./models_cache/roberta-base')
    ap.add_argument('--eval_dir', default='./data/wdc_lspm_sampled/eval')
    ap.add_argument('--kb_path', default='./data/wdc_lspm_sampled/wdc_products_kb.jsonl')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--max_tables', type=int, default=8000)
    ap.add_argument('--out', default='./results/qual_dump_rsupcon.jsonl')
    args = ap.parse_args()
    dev = torch.device(args.device)

    print(f"[load] model={args.model_name} ckpt={args.ckpt}", flush=True)
    model = RSupConModel(model_name=args.model_name).to(dev)
    state = torch.load(args.ckpt, map_location=dev, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state: state = state['model_state_dict']
    if isinstance(state, dict) and 'model' in state and 'criterion' not in state: state = state['model']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    m, u = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(m)} unexpected={len(u)}", flush=True)
    model.eval()
    tok = model.tokenizer

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

    ent_texts = [f"[COL] name [VAL] {' '.join(str(n).split(' ')[:60])}".strip() for n in kb_names]
    t0 = time.time()
    kb_emb = encode(model, tok, ent_texts, dev, bs=args.bs, desc="kb_enc")
    kb_emb = kb_emb.to(dev)
    print(f"[KB] shape={tuple(kb_emb.shape)} time={(time.time()-t0)/60:.2f}min", flush=True)

    # ---------- eval cells ----------
    print(f"[Eval] loading {args.eval_dir}", flush=True)
    ds = TableAlignmentDataset(root=args.eval_dir)
    n_lim = min(args.max_tables, len(ds))
    print(f"[Eval] tables={len(ds)}, dumping up to {n_lim}", flush=True)

    queries = []  # list of (table_idx, cell_idx, cell_text, gold_id)
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

    cell_texts = [f"[COL] title [VAL] {' '.join(q[2].split(' ')[:50])}".strip() for q in queries]
    t1 = time.time()
    q_emb = encode(model, tok, cell_texts, dev, bs=args.bs, desc="q_enc")
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
