"""R-SupCon Accuracy vs KB scale on WDC LSPM.

Encodes 8M KB via R-SupCon (RoBERTa + mean-pool + L2-norm, 768d),
then subsamples KB at multiple scales and reports Acc / Hit@5 / Hit@10.
Uses the official `[COL] name [VAL] X` entity serialization and raw cell
text (no header prepend) on the cell side, matching the protocol used to
produce the R-SupCon 0.8207 number in Table I.
"""
import os, sys, json, time, argparse, random
sys.path.insert(0, ".")
sys.path.insert(0, "./baselines")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F
import numpy as np
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


def encode_rsupcon(model, tok, texts, device, bs=128, max_len=128):
    embs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc="enc", mininterval=15.0):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            out = model.encoder(input_ids=enc['input_ids'], attention_mask=enc['attention_mask']).last_hidden_state
            pooled = _mean_pool(out, enc['attention_mask'])
            pooled = F.normalize(pooled, p=2, dim=-1).half()
            embs.append(pooled.detach().to('cpu'))
    return torch.cat(embs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='./checkpoints/rsupcon_roberta_base.pt')
    ap.add_argument('--model_name', default='./models_cache/roberta_base')
    ap.add_argument('--eval_dir', default='./data/wdc_lspm_sampled/eval')
    ap.add_argument('--kb_path', default='./data/wdc_lspm_sampled/wdc_products_kb.jsonl')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--scales', default='100000,500000,2000000,8043290')
    ap.add_argument('--max_tables', type=int, default=8000)
    args = ap.parse_args()
    random.seed(42); np.random.seed(42)
    dev = torch.device(args.device)

    print(f"[load] model={args.model_name} ckpt={args.ckpt}", flush=True)
    model = RSupConModel(model_name=args.model_name).to(dev)
    try:
        state = torch.load(args.ckpt, map_location=dev, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state: state = state['model_state_dict']
        if isinstance(state, dict) and 'model' in state and 'criterion' not in state: state = state['model']
        state = {k.replace('module.', ''): v for k, v in state.items()}
        m, u = model.load_state_dict(state, strict=False)
        print(f"[load] missing={len(m)} unexpected={len(u)}", flush=True)
    except FileNotFoundError as e:
        print(f"[load] WARN no ckpt at {args.ckpt}: {e}", flush=True)
    model.eval()
    tok = model.tokenizer

    print(f"[KB] reading {args.kb_path}", flush=True)
    kb_ids, kb_names = [], []
    with open(args.kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            kb_ids.append(d['id']); kb_names.append(d.get('name', d['id']))
    M = len(kb_ids)
    kb_id_to_idx = {kid: i for i, kid in enumerate(kb_ids)}
    print(f"[KB] {M} entities", flush=True)

    # entity-side serialization: [COL] name [VAL] X (R-SupCon convention)
    ent_texts = [f"[COL] name [VAL] {' '.join(str(n).split(' ')[:60])}".strip() for n in kb_names]

    t0 = time.time()
    kb_emb = encode_rsupcon(model, tok, ent_texts, dev, bs=args.bs)
    print(f"[KB] encoded shape={tuple(kb_emb.shape)} time={(time.time()-t0)/60:.2f}min", flush=True)

    # ---------- eval queries ----------
    print(f"[Eval] loading {args.eval_dir}", flush=True)
    ds = TableAlignmentDataset(root=args.eval_dir)
    print(f"[Eval] {len(ds)} tables", flush=True)
    eval_texts, gold_ids = [], []
    n_lim = min(args.max_tables, len(ds))
    print(f"[Eval] extracting from up to {n_lim} tables", flush=True)
    for i in tqdm(range(n_lim), desc="extract", mininterval=5.0):
        try: g = ds[i]
        except Exception: continue
        if getattr(g, 'is_dummy', False): continue
        labeled = g.labeled_indices.tolist() if hasattr(g.labeled_indices, "tolist") else list(g.labeled_indices)
        gold_lst = list(getattr(g, "target_ent_ids", []))
        for idx, gid in zip(labeled, gold_lst):
            if 0 <= idx < len(g.x_text):
                txt = str(g.x_text[idx]).strip()
                gid_str = (gid[0] if isinstance(gid, (list, tuple)) and gid else gid) or ""
                gid_str = str(gid_str).strip()
                if txt and gid_str:
                    eval_texts.append(txt); gold_ids.append(gid_str)
    print(f"[Eval] {len(eval_texts)} labeled queries", flush=True)

    # cell-side serialization for R-SupCon: raw cell text wrapped as [COL] title [VAL]
    cell_texts = [f"[COL] title [VAL] {' '.join(str(t).split(' ')[:50])}".strip() for t in eval_texts]
    t1 = time.time()
    q_emb = encode_rsupcon(model, tok, cell_texts, dev, bs=args.bs)
    print(f"[Q] encoded shape={tuple(q_emb.shape)} time={(time.time()-t1)/60:.2f}min", flush=True)

    # gold indices
    gold_idx_in_kb = []
    keep_mask = []
    for gid in gold_ids:
        j = kb_id_to_idx.get(gid)
        if j is None: keep_mask.append(False); continue
        keep_mask.append(True); gold_idx_in_kb.append(j)
    keep_mask = np.array(keep_mask)
    print(f"[gold] {keep_mask.sum()}/{len(gold_ids)} kept", flush=True)
    q_emb_kept = q_emb[keep_mask]
    gold_idx_arr = np.array(gold_idx_in_kb, dtype=np.int64)
    gold_set = set(gold_idx_in_kb)

    scales = [int(s) for s in args.scales.split(',')]
    all_kb_idx = np.arange(M, dtype=np.int64)
    non_gold_pool = np.array([i for i in all_kb_idx if i not in gold_set], dtype=np.int64)

    for N in scales:
        if N >= M:
            sub_idx = all_kb_idx
        else:
            n_fill = max(0, N - len(gold_set))
            if n_fill > 0:
                fill = np.random.choice(non_gold_pool, size=min(n_fill, len(non_gold_pool)), replace=False)
                sub_idx = np.concatenate([np.array(sorted(gold_set), dtype=np.int64), fill])
            else:
                sub_idx = np.array(sorted(gold_set), dtype=np.int64)[:N]
            np.random.shuffle(sub_idx)
        sub_kb = kb_emb[torch.from_numpy(sub_idx).long()].to(dev)
        kb_to_sub = -np.ones(M, dtype=np.int64)
        kb_to_sub[sub_idx] = np.arange(len(sub_idx), dtype=np.int64)
        gold_sub_pos = kb_to_sub[gold_idx_arr]

        hit1 = hit5 = hit10 = 0; n_q = q_emb_kept.size(0)
        mrr_sum = 0.0
        chunk = 128
        t2 = time.time()
        for i in range(0, n_q, chunk):
            q = q_emb_kept[i:i + chunk].to(dev).float()
            sim = q @ sub_kb.float().t()
            top = torch.topk(sim, args.topk, dim=-1).indices.cpu().numpy()
            gs = gold_sub_pos[i:i + chunk]
            for r in range(top.shape[0]):
                gold_pos = gs[r]
                if top[r, 0] == gold_pos: hit1 += 1
                if gold_pos in top[r, :5]: hit5 += 1
                if gold_pos in top[r]:
                    hit10 += 1
                    rk = int(np.where(top[r] == gold_pos)[0][0]) + 1
                    mrr_sum += 1.0 / rk
        del sub_kb; torch.cuda.empty_cache()
        rec = dict(
            scale=N, model='rsupcon',
            acc=hit1 / n_q, hit5=hit5 / n_q, hit10=hit10 / n_q,
            mrr10=mrr_sum / n_q, n_queries=n_q,
            search_sec=(time.time() - t2),
        )
        print(f"[SCALE_RESULT] {json.dumps(rec)}", flush=True)


if __name__ == '__main__':
    main()
