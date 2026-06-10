"""HyperTabAlign Accuracy vs KB scale on WDC LSPM.

Pipeline:
  1) load HyperTab ckpt, encode 8M KB (FP16, 256-d).
  2) iterate eval set, encode every labeled cell (direct path, no GNN).
  3) for each KB scale in {100k, 500k, 2M, 8M}, subsample = all_gold + random fill,
     run cosine top-10 on subset, report Acc / Hit@5 / Hit@10.

Output: prints SCALE_RESULTS json line per scale.
"""
import os, sys, json, time, argparse, random
sys.path.insert(0, ".")
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
from models.unified_model import HyperGraphRAGModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='./checkpoints/wdc_sota_v2.pt')
    ap.add_argument('--eval_dir', default='./data/wdc_lspm_sampled/eval')
    ap.add_argument('--kb_path', default='./data/wdc_lspm_sampled/wdc_products_kb.jsonl')
    ap.add_argument('--model_name', default='./models_cache/deberta-v3-base')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--kb_bs', type=int, default=256)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--scales', default='100000,500000,2000000,8043290')
    ap.add_argument('--max_tables', type=int, default=8000)
    args = ap.parse_args()
    random.seed(42); np.random.seed(42)
    dev = torch.device(args.device)

    print(f"[load] ckpt={args.ckpt}", flush=True)
    config = dict(model_name=args.model_name, gnn_layers=2, gnn_hidden_dim=768, retrieval_dim=256)
    model = HyperGraphRAGModel(config).to(dev)
    state = torch.load(args.ckpt, map_location=dev, weights_only=False)
    if isinstance(state, dict) and 'model' in state: state = state['model']
    elif isinstance(state, dict) and 'model_state_dict' in state: state = state['model_state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    m, u = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(m)} unexpected={len(u)}", flush=True)
    model.eval()

    print(f"[KB] reading {args.kb_path}", flush=True)
    kb_ids, kb_names = [], []
    with open(args.kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            kb_ids.append(d['id']); kb_names.append(d.get('name', d['id']))
    M = len(kb_ids)
    kb_id_to_idx = {kid: i for i, kid in enumerate(kb_ids)}
    print(f"[KB] {M} entities", flush=True)

    t0 = time.time()
    chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, M, args.kb_bs), desc="kb_enc", mininterval=15.0):
            names = kb_names[i:i + args.kb_bs]; paths = [''] * len(names)
            kb_e = model(graph=None, mode='encode_kb', names=names, paths=paths)
            kb_e_proj = model.retriever.shared_proj(kb_e)
            kb_e_proj = F.normalize(kb_e_proj, p=2, dim=-1).half()
            chunks.append(kb_e_proj.detach().to('cpu'))
    kb_emb = torch.cat(chunks, dim=0)
    print(f"[KB] encoded shape={tuple(kb_emb.shape)} dtype={kb_emb.dtype} time={(time.time()-t0)/60:.2f}min", flush=True)

    # ---------- eval queries via the SAME direct path used by KB encoder ----------
    print(f"[Eval] loading dataset from {args.eval_dir}", flush=True)
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

    # encode queries via SAME path the KB used (direct mode = encode_kb on cell text)
    q_chunks = []
    t1 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_texts), args.kb_bs), desc="q_enc", mininterval=10.0):
            texts = eval_texts[i:i + args.kb_bs]; paths = [''] * len(texts)
            e = model(graph=None, mode='encode_kb', names=texts, paths=paths)
            e_proj = model.retriever.shared_proj(e)
            e_proj = F.normalize(e_proj, p=2, dim=-1).half()
            q_chunks.append(e_proj.detach().to('cpu'))
    q_emb = torch.cat(q_chunks, dim=0)
    print(f"[Q] encoded shape={tuple(q_emb.shape)} time={(time.time()-t1)/60:.2f}min", flush=True)

    # ---------- gold indices ----------
    gold_idx_in_kb = []
    drop = 0
    for gid in gold_ids:
        j = kb_id_to_idx.get(gid)
        if j is None: drop += 1; continue
        gold_idx_in_kb.append(j)
    keep_mask = np.array([kb_id_to_idx.get(g) is not None for g in gold_ids])
    print(f"[gold] {keep_mask.sum()}/{len(gold_ids)} queries have gold in full KB (dropped {drop})", flush=True)

    gold_set = set(gold_idx_in_kb)
    print(f"[gold] {len(gold_set)} unique gold KB indices", flush=True)

    # filter q_emb + gold to queries whose gold is in full KB
    q_emb_kept = q_emb[keep_mask]
    gold_idx_arr = np.array(gold_idx_in_kb, dtype=np.int64)  # one per kept query

    # ---------- scale loop ----------
    scales = [int(s) for s in args.scales.split(',')]
    all_kb_idx = np.arange(M, dtype=np.int64)
    non_gold_pool = np.array([i for i in all_kb_idx if i not in gold_set], dtype=np.int64)
    print(f"[pool] non-gold pool size = {len(non_gold_pool)}", flush=True)

    print(f"[SCALE] starting scale loop over {scales}", flush=True)
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
        # build sub index on GPU
        sub_kb = kb_emb[torch.from_numpy(sub_idx).long()].to(dev)
        # remap gold idx -> position in sub_idx
        kb_to_sub = -np.ones(M, dtype=np.int64)
        kb_to_sub[sub_idx] = np.arange(len(sub_idx), dtype=np.int64)
        gold_sub_pos = kb_to_sub[gold_idx_arr]  # one per kept query, all >= 0 since gold included

        # search in chunks
        hit1 = hit5 = hit10 = 0; n_q = q_emb_kept.size(0)
        mrr_sum = 0.0
        chunk = 256
        t2 = time.time()
        for i in range(0, n_q, chunk):
            q = q_emb_kept[i:i + chunk].to(dev).float()
            sim = q @ sub_kb.float().t()  # [chunk, |sub|]
            top = torch.topk(sim, args.topk, dim=-1).indices.cpu().numpy()  # [chunk, K]
            gs = gold_sub_pos[i:i + chunk]  # [chunk]
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
            scale=N, model='hypertab',
            acc=hit1 / n_q, hit5=hit5 / n_q, hit10=hit10 / n_q,
            mrr10=mrr_sum / n_q, n_queries=n_q,
            search_sec=(time.time() - t2),
        )
        print(f"[SCALE_RESULT] {json.dumps(rec)}", flush=True)


if __name__ == '__main__':
    main()
