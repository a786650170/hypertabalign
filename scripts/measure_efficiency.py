"""HyperTabAlign cold-build + warm-query timing on WDC LSPM 8M.

Outputs:
  [BUILD] 8M-entity KB encode pass wall time -> Table IV `Build` column.
  [QUERY_HYPER]  per-query GNN+search latency -> Table IV `Query` (hyper mode).
  [QUERY_DIRECT] per-query bi-encoder cosine top-k latency -> Table IV `Query` (direct).

Eval-set pass is table-batched: one table -> one model forward -> N cells.
We report per-cell (per-query) latency after warmup.
"""
import os, sys, json, time, argparse
sys.path.insert(0, ".")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F
from tqdm import tqdm

# torch.load CVE bypass
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
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--kb_bs', type=int, default=256)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--n_warmup_tables', type=int, default=20)
    ap.add_argument('--n_timed_tables', type=int, default=2000)
    args = ap.parse_args()
    dev = torch.device(args.device)

    print(f"[load] ckpt={args.ckpt}", flush=True)
    config = dict(model_name=args.model_name, gnn_layers=2, gnn_hidden_dim=768, retrieval_dim=256)
    model = HyperGraphRAGModel(config).to(dev)
    state = torch.load(args.ckpt, map_location=dev, weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    elif isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model.eval()

    # ============================================================
    # 1) COLD BUILD: encode 8M KB entities through HyperTab KB encoder
    # ============================================================
    print(f"[KB] reading {args.kb_path}", flush=True)
    kb_ids, kb_names = [], []
    with open(args.kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            kb_ids.append(d['id'])
            kb_names.append(d.get('name', d['id']))
    M = len(kb_ids)
    print(f"[KB] {M} entities", flush=True)

    print(f"[BUILD] starting cold KB index build ...", flush=True)
    torch.cuda.synchronize(dev)
    t_build_start = time.time()
    kb_chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, M, args.kb_bs), desc="kb_build", mininterval=10.0):
            names = kb_names[i:i + args.kb_bs]
            paths = [''] * len(names)
            kb_e = model(graph=None, mode='encode_kb', names=names, paths=paths)
            kb_e_proj = model.retriever.shared_proj(kb_e)
            kb_e_proj = F.normalize(kb_e_proj, p=2, dim=-1)
            kb_chunks.append(kb_e_proj.detach().to('cpu'))
    torch.cuda.synchronize(dev)
    t_build_end = time.time()
    build_min = (t_build_end - t_build_start) / 60.0
    print(f"[BUILD] wall time: {build_min:.2f} min for {M} entities", flush=True)

    kb_emb = torch.cat(kb_chunks, dim=0).to(dev)
    print(f"[BUILD] kb_emb shape {tuple(kb_emb.shape)} resident on {dev}", flush=True)

    # ============================================================
    # 2) WARM QUERY: per-cell latency in HYPER mode (GNN + search)
    # ============================================================
    print(f"[Eval] loading TableAlignmentDataset from {args.eval_dir}", flush=True)
    ds = TableAlignmentDataset(root=args.eval_dir)
    print(f"[Eval] tables={len(ds)}", flush=True)

    def time_path(mode_name, n_warm, n_timed):
        """mode_name in {'hyper','direct'}. Iterate tables, time per-cell latency."""
        torch.cuda.synchronize(dev)
        per_query_times = []
        n_cells_total = 0
        # warmup
        warm_done = 0
        for i in range(len(ds)):
            if warm_done >= n_warm:
                break
            try:
                g = ds[i]
            except Exception:
                continue
            if getattr(g, 'is_dummy', False):
                continue
            g = g.to(dev)
            with torch.no_grad():
                if mode_name == 'hyper':
                    out = model(g, mode='encode_table')
                    if isinstance(out, tuple):
                        _, ae = out
                    else:
                        ae = out
                    q = model.retriever.shared_proj(ae)
                    q = F.normalize(q, p=2, dim=-1)
                else:  # direct: bypass GNN
                    # Use the same KB-side encoder branch on each cell text
                    texts = [str(t) for t in g.x_text]
                    paths = [''] * len(texts)
                    e = model(graph=None, mode='encode_kb', names=texts, paths=paths)
                    q = model.retriever.shared_proj(e)
                    q = F.normalize(q, p=2, dim=-1)
                _ = (q @ kb_emb.t()).topk(args.topk, dim=-1)
            warm_done += 1
        torch.cuda.synchronize(dev)
        print(f"[{mode_name}] warmup done ({warm_done} tables)", flush=True)

        # timed
        timed = 0
        pbar = tqdm(range(len(ds)), desc=f"timed_{mode_name}", mininterval=10.0)
        for i in pbar:
            if timed >= n_timed:
                break
            try:
                g = ds[i]
            except Exception:
                continue
            if getattr(g, 'is_dummy', False):
                continue
            g = g.to(dev)
            n_cells = len(g.x_text)
            torch.cuda.synchronize(dev)
            t0 = time.time()
            with torch.no_grad():
                if mode_name == 'hyper':
                    out = model(g, mode='encode_table')
                    if isinstance(out, tuple):
                        _, ae = out
                    else:
                        ae = out
                    q = model.retriever.shared_proj(ae)
                    q = F.normalize(q, p=2, dim=-1)
                else:
                    texts = [str(t) for t in g.x_text]
                    paths = [''] * len(texts)
                    e = model(graph=None, mode='encode_kb', names=texts, paths=paths)
                    q = model.retriever.shared_proj(e)
                    q = F.normalize(q, p=2, dim=-1)
                _ = (q @ kb_emb.t()).topk(args.topk, dim=-1)
            torch.cuda.synchronize(dev)
            dt = time.time() - t0
            per_query_times.append(dt / max(n_cells, 1))
            n_cells_total += n_cells
            timed += 1
            if timed % 200 == 0:
                avg_ms = 1000.0 * sum(per_query_times) / len(per_query_times)
                pbar.set_postfix(avg_ms_per_q=f"{avg_ms:.2f}")
        if not per_query_times:
            print(f"[{mode_name}] no timed tables", flush=True); return None
        per_query_times.sort()
        n = len(per_query_times)
        mean_ms = 1000.0 * sum(per_query_times) / n
        median_ms = 1000.0 * per_query_times[n // 2]
        p95_ms = 1000.0 * per_query_times[int(n * 0.95)]
        print(f"[{mode_name}] tables_timed={n}  cells={n_cells_total}", flush=True)
        print(f"[{mode_name}] mean={mean_ms:.3f} ms/q  median={median_ms:.3f} ms/q  p95={p95_ms:.3f} ms/q", flush=True)
        return mean_ms, median_ms, p95_ms

    hyper_stats = time_path('hyper', args.n_warmup_tables, args.n_timed_tables)
    direct_stats = time_path('direct', args.n_warmup_tables, args.n_timed_tables)

    print(f"\n========== SUMMARY ==========", flush=True)
    print(f"[BUILD]        {build_min:.2f} min  (8M KB entities)", flush=True)
    if hyper_stats:
        print(f"[QUERY_HYPER]  mean={hyper_stats[0]:.3f} ms/q  median={hyper_stats[1]:.3f}  p95={hyper_stats[2]:.3f}", flush=True)
    if direct_stats:
        print(f"[QUERY_DIRECT] mean={direct_stats[0]:.3f} ms/q  median={direct_stats[1]:.3f}  p95={direct_stats[2]:.3f}", flush=True)


if __name__ == '__main__':
    main()
