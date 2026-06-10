"""Per-cell qualitative dump for HyperTabAlign on WDC LSPM 8M.

Output one JSONL line per labeled eval cell with:
  cell_text, table_idx, table_headers, table_neighbors_in_row,
  gold_id, gold_name,
  top1_id, top1_name, top1_score, top5_ids, top5_names,
  alpha_row, alpha_col, gamma

Supports paper §V qualitative case table and gate-weight distribution figure.
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
from models.unified_model import HyperGraphRAGModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='./checkpoints/wdc_sota_v2.pt')
    ap.add_argument('--eval_dir', default='./data/wdc_lspm_sampled/eval')
    ap.add_argument('--kb_path', default='./data/wdc_lspm_sampled/wdc_products_kb.jsonl')
    ap.add_argument('--model_name', default='./models_cache/deberta-v3-base')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--kb_bs', type=int, default=256)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--max_tables', type=int, default=8000)
    ap.add_argument('--out', default='./results/qual_dump_hypertab.jsonl')
    args = ap.parse_args()
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

    # ---------- gamma ----------
    gamma_scalar = None
    for n, p in model.named_parameters():
        if 'lambda' in n.lower() or 'gate_lambda' in n.lower() or 'residual_lambda' in n.lower():
            gamma_scalar = float(torch.sigmoid(p).item()) if p.numel() == 1 else None
            print(f"[gamma] from param '{n}': lambda={float(p.item()):.4f}  sigmoid={gamma_scalar}", flush=True)
            break
    if gamma_scalar is None:
        # fallback: search by typical name
        for n, p in model.named_parameters():
            if p.numel() == 1 and 'lambda' in n.lower():
                gamma_scalar = float(torch.sigmoid(p).item())
                print(f"[gamma] fallback from '{n}': sigmoid={gamma_scalar}", flush=True)
                break

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
    kb_chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, M, args.kb_bs), desc="kb_enc", mininterval=15.0):
            names = kb_names[i:i + args.kb_bs]; paths = [''] * len(names)
            kb_e = model(graph=None, mode='encode_kb', names=names, paths=paths)
            kb_e_proj = model.retriever.shared_proj(kb_e)
            kb_e_proj = F.normalize(kb_e_proj, p=2, dim=-1).half()
            kb_chunks.append(kb_e_proj.detach().to('cpu'))
    kb_emb = torch.cat(kb_chunks, dim=0).to(dev)
    print(f"[KB] shape={tuple(kb_emb.shape)} time={(time.time()-t0)/60:.2f}min", flush=True)

    # ---------- iterate eval tables, dump per labeled cell ----------
    print(f"[Eval] loading {args.eval_dir}", flush=True)
    ds = TableAlignmentDataset(root=args.eval_dir)
    print(f"[Eval] {len(ds)} tables", flush=True)

    n_lim = min(args.max_tables, len(ds))
    out = open(args.out, 'w', encoding='utf-8')
    n_dumped = 0
    n_alpha_dumped = 0
    pbar = tqdm(range(n_lim), desc="dump", mininterval=10.0)
    for ti in pbar:
        try: g = ds[ti]
        except Exception: continue
        if getattr(g, 'is_dummy', False): continue
        n_cells = len(g.x_text)
        if n_cells == 0: continue

        # encode the table once via encode_table to get alpha + retrieval embedding
        try:
            with torch.no_grad():
                out_pair = model(g.to(dev), mode='encode_table')
                if isinstance(out_pair, tuple):
                    _, align_emb = out_pair
                else:
                    align_emb = out_pair
                q_proj = model.retriever.shared_proj(align_emb)
                q_proj = F.normalize(q_proj, p=2, dim=-1)  # [N_cells, 256]
        except Exception as e:
            continue

        # try to fetch alpha (row/col gate) from any registered buffer / attribute
        alpha = None
        for attr in ['last_alpha', 'alpha_row_col', 'last_gate_alpha']:
            if hasattr(model, attr):
                a = getattr(model, attr)
                if a is not None:
                    alpha = a.detach().to('cpu').tolist() if isinstance(a, torch.Tensor) else a
                    break
        # alpha is per-node [N_nodes, 2]; align with labeled cell ids below

        # search top-K against full KB
        with torch.no_grad():
            sim = q_proj.float() @ kb_emb.float().t()  # [N_cells, M]
            top_vals, top_inds = torch.topk(sim, args.topk, dim=-1)
        top_inds_cpu = top_inds.detach().cpu().tolist()
        top_vals_cpu = top_vals.detach().cpu().tolist()

        labeled = g.labeled_indices.tolist() if hasattr(g.labeled_indices, 'tolist') else list(g.labeled_indices)
        gold_lst = list(getattr(g, 'target_ent_ids', []))

        # capture table headers + row context for selected cells
        texts = [str(t) for t in g.x_text]
        # coordinates if available
        coords = None
        if hasattr(g, 'coords'):
            try: coords = g.coords.tolist() if isinstance(g.coords, torch.Tensor) else list(g.coords)
            except Exception: coords = None

        for idx, gid in zip(labeled, gold_lst):
            if not (0 <= idx < n_cells): continue
            gid_str = (gid[0] if isinstance(gid, (list, tuple)) and gid else gid) or ''
            gid_str = str(gid_str).strip()
            if not gid_str: continue
            gold_idx = kb_id_to_idx.get(gid_str)
            gold_name = kb_names[gold_idx] if gold_idx is not None else None

            t1_kb = top_inds_cpu[idx][0]
            t1_id = kb_ids[t1_kb]; t1_name = kb_names[t1_kb]; t1_score = float(top_vals_cpu[idx][0])
            top5_ids = [kb_ids[j] for j in top_inds_cpu[idx][:args.topk]]
            top5_names = [kb_names[j] for j in top_inds_cpu[idx][:args.topk]]

            # row context: cells in same row as this one (via coords if avail)
            row_ctx = None
            if coords is not None and idx < len(coords):
                try:
                    r_self = coords[idx][0]
                    row_ctx = [texts[k] for k in range(n_cells)
                               if 0 <= k < len(coords) and coords[k][0] == r_self and k != idx]
                except Exception:
                    row_ctx = None

            a_row = a_col = None
            if alpha is not None and idx < len(alpha):
                try:
                    a_row = float(alpha[idx][0]); a_col = float(alpha[idx][1])
                    n_alpha_dumped += 1
                except Exception:
                    pass

            rec = dict(
                table_idx=ti, cell_idx=int(idx), n_cells=n_cells,
                cell_text=texts[idx],
                row_context=row_ctx,
                gold_id=gid_str, gold_name=gold_name,
                top1_id=t1_id, top1_name=t1_name, top1_score=t1_score,
                top5_ids=top5_ids, top5_names=top5_names,
                alpha_row=a_row, alpha_col=a_col,
                gamma=gamma_scalar,
            )
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_dumped += 1
        if n_dumped % 1000 < 5:
            pbar.set_postfix(dumped=n_dumped, with_alpha=n_alpha_dumped)
    out.close()
    print(f"[done] dumped {n_dumped} labeled cells, {n_alpha_dumped} with alpha -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
