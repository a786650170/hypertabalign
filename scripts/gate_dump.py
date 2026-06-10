"""Fast per-cell gate weight dump for HyperTabAlign.

Hooks the row/column fusion gate (`rc_gates` modules in models/encoder/table_gnn.py:222)
and captures alpha = softmax(linear(concat([h_row, h_col]))) per node, per layer.

No KB encoding needed -- only the table-side GNN forward. Fast (~5-10 min for 2000 tables).
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
    ap.add_argument('--model_name', default='./models_cache/deberta-v3-base')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n_tables', type=int, default=2000)
    ap.add_argument('--out', default='./results/gate_alpha_dump.jsonl')
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

    # ---- locate rc_gates list ----
    rc_gates_owner = None
    rc_gates = None
    for name, module in model.named_modules():
        if hasattr(module, 'rc_gates') and isinstance(module.rc_gates, torch.nn.ModuleList):
            rc_gates_owner = (name, module)
            rc_gates = module.rc_gates
            break
    if rc_gates is None:
        print("[err] could not find rc_gates", flush=True); sys.exit(1)
    print(f"[hook] rc_gates owner={rc_gates_owner[0]} n_layers={len(rc_gates)}", flush=True)

    # ---- gamma ----
    gnn_gate_lambda = None
    if hasattr(rc_gates_owner[1], 'gnn_gate_logit'):
        gnn_gate_lambda = float(rc_gates_owner[1].gnn_gate_logit.item())
        gamma = float(torch.sigmoid(torch.tensor(gnn_gate_lambda)).item())
        print(f"[gamma] lambda={gnn_gate_lambda:.4f}  sigmoid->gamma={gamma:.4f}", flush=True)
    else:
        gamma = None
        for n, p in model.named_parameters():
            if 'gate_logit' in n and p.numel() == 1:
                gnn_gate_lambda = float(p.item())
                gamma = float(torch.sigmoid(p).item())
                print(f"[gamma] from '{n}': lambda={gnn_gate_lambda:.4f}  gamma={gamma:.4f}", flush=True)
                break

    # ---- register hooks ----
    captures = {}  # layer_idx -> [N_nodes, 2] tensor (alpha after softmax)
    def make_hook(idx):
        def hook(mod, inp, out):
            # out is logits over {row, col}; apply softmax to get alpha
            captures[idx] = torch.softmax(out, dim=-1).detach().cpu()
        return hook
    handles = []
    for i, g in enumerate(rc_gates):
        handles.append(g.register_forward_hook(make_hook(i)))

    # ---- iterate eval tables ----
    print(f"[Eval] loading {args.eval_dir}", flush=True)
    ds = TableAlignmentDataset(root=args.eval_dir)
    n_total = len(ds)
    n_lim = min(args.n_tables, n_total)
    print(f"[Eval] {n_total} tables, dumping up to {n_lim}", flush=True)

    out = open(args.out, 'w', encoding='utf-8')
    n_dumped = 0; n_skipped = 0
    pbar = tqdm(range(n_lim), desc="dump", mininterval=5.0)
    t0 = time.time()
    for ti in pbar:
        try: g = ds[ti]
        except Exception: n_skipped += 1; continue
        if getattr(g, 'is_dummy', False): n_skipped += 1; continue
        n_cells = len(g.x_text)
        if n_cells == 0: continue
        captures.clear()
        try:
            with torch.no_grad():
                _ = model(g.to(dev), mode='encode_table')
        except Exception as e:
            n_skipped += 1; continue
        if not captures:
            n_skipped += 1; continue

        # collect alpha per node across layers
        layers = sorted(captures.keys())
        alpha_per_layer = {L: captures[L].tolist() for L in layers}  # each: [N_nodes, 2]
        coords = None
        if hasattr(g, 'coords'):
            try:
                coords = g.coords.tolist() if isinstance(g.coords, torch.Tensor) else list(g.coords)
            except Exception:
                coords = None
        texts = [str(t) for t in g.x_text]

        for n in range(n_cells):
            is_header = (coords[n][0] == -1) if coords and n < len(coords) else False
            rec = dict(
                table_idx=ti, node_idx=n, n_cells=n_cells,
                is_header=is_header,
                cell_text=texts[n][:120],
                alpha_per_layer={L: alpha_per_layer[L][n] if n < len(alpha_per_layer[L]) else None for L in layers},
                gamma=gamma,
            )
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_dumped += 1
        if ti % 100 == 0:
            elapsed = time.time() - t0
            pbar.set_postfix(dumped=n_dumped, sec=int(elapsed))
            sys.stdout.flush()
    out.close()
    for h in handles: h.remove()
    print(f"[done] dumped {n_dumped} nodes from {n_lim - n_skipped} tables (skipped {n_skipped}) -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
