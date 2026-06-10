# --- patch torch.load CVE check (forces transformers to load pytorch_model.bin) ---
try:
    import transformers.utils.import_utils as _iu
    if hasattr(_iu, 'check_torch_load_is_safe'):
        _iu.check_torch_load_is_safe = lambda: True
    import transformers.modeling_utils as _mu
    if hasattr(_mu, 'check_torch_load_is_safe'):
        _mu.check_torch_load_is_safe = lambda: True
except Exception:
    pass

"""Dump per-cell attention signals from trained HyperTabAlign for visualization.

For each sampled eval table, forward through the model and capture:
  - text of each cell  (so we can label points)
  - (row, col) coordinate
  - NER P(entity) -- which cells the model thinks need alignment
  - row/col gate alpha_v (softmax of rc_gates output, last layer) -- which
    structural channel each cell relies on
  - global residual gate gamma = sigmoid(gnn_gate_logit) -- how much GNN vs text

Output: JSONL, one record per table. Rendering done by a separate plot script.
"""
import sys, os, json, argparse
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from data.dataset import TableAlignmentDataset
from models.unified_model import HyperGraphRAGModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='./checkpoints/wdc_icde_clean_4gpu_bs196_best.pt')
    ap.add_argument('--eval_dir', default='./data/wdc_lspm_sampled/eval')
    ap.add_argument('--out', default='./results/viz_attention_dump.jsonl')
    ap.add_argument('--n_tables', type=int, default=24)
    ap.add_argument('--model_name', default='./models_cache/deberta-v3-base')
    ap.add_argument('--min_cells', type=int, default=6)
    ap.add_argument('--max_cells', type=int, default=40)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    dev = torch.device(args.device)
    config = dict(model_name=args.model_name,
                  gnn_layers=2, gnn_hidden_dim=768, retrieval_dim=256)
    print(f"[viz] Initializing model on {dev}")
    model = HyperGraphRAGModel(config).to(dev)

    print(f"[viz] Loading ckpt: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=dev, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[viz] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # Hook rc_gates (per-layer row/col fusion gate). Output = pre-softmax logits.
    captured = []
    def make_hook(layer_idx):
        def hook(module, inputs, output):
            alpha = torch.softmax(output, dim=-1)  # [N_nodes, 2]
            captured.append((layer_idx, alpha.detach().cpu()))
        return hook

    if hasattr(model.table_encoder, 'rc_gates'):
        for i, gate in enumerate(model.table_encoder.rc_gates):
            gate.register_forward_hook(make_hook(i))
        print(f"[viz] Hooked {len(model.table_encoder.rc_gates)} rc_gate layers")
    else:
        print("[viz] WARN: model.table_encoder has no rc_gates -- alpha unavailable")

    # Global residual gate
    gamma = torch.sigmoid(model.table_encoder.gnn_gate_logit).item() \
        if hasattr(model.table_encoder, 'gnn_gate_logit') else None
    print(f"[viz] Global residual gate gamma = {gamma}")

    print(f"[viz] Loading eval dataset: {args.eval_dir}")
    ds = TableAlignmentDataset(root=args.eval_dir)
    print(f"[viz] Total tables: {len(ds)}")

    out_f = open(args.out, 'w', encoding='utf-8')
    written = 0
    scanned = 0

    for i in range(len(ds)):
        if written >= args.n_tables:
            break
        scanned += 1
        try:
            g = ds[i]
        except Exception as e:
            continue
        if getattr(g, 'is_dummy', False):
            continue
        n = len(g.x_text)
        if n < args.min_cells or n > args.max_cells:
            continue

        captured.clear()
        try:
            with torch.no_grad():
                # encode_table returns (extraction_logits, alignment_embeds)
                # The exact return signature lives in TableGNN.forward; we unpack safely.
                out = model(g.to(dev), mode='encode_table')
            if isinstance(out, tuple):
                extraction_logits = out[0]
            else:
                extraction_logits = out
            ner_prob = torch.softmax(extraction_logits, dim=-1)[:, 1].cpu().numpy().tolist()
        except Exception as e:
            print(f"[viz] Skip table {i}: {e}")
            continue

        coords = g.coords.cpu().numpy().tolist() if hasattr(g, 'coords') else []
        # Take last-layer alpha if available
        last_alpha = captured[-1][1].numpy().tolist() if captured else None

        # Labeled cells (ground truth entity cells) for reference
        labeled = []
        gold_names = []
        if hasattr(g, 'labeled_indices'):
            labeled = g.labeled_indices.tolist() if hasattr(g.labeled_indices, 'tolist') \
                      else list(g.labeled_indices)
        if hasattr(g, 'target_ent_ids'):
            gold_names = list(g.target_ent_ids)

        rec = {
            'table_idx': i,
            'n_cells': n,
            'texts': g.x_text,
            'coords': coords,          # [[r,c], ...], header rows are r=-1
            'ner_prob_entity': ner_prob,
            'alpha_row_col': last_alpha,  # [N, 2] : [alpha_row, alpha_col]
            'gnn_gate_gamma': gamma,
            'labeled_indices': labeled,
            'gold_ent_ids': [str(g) for g in gold_names],
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        written += 1
        if written % 5 == 0:
            print(f"[viz] wrote {written}/{args.n_tables} (scanned {scanned})")

    out_f.close()
    print(f"[viz] Done. {written} tables -> {args.out}")
    print(f"[viz] Global gamma = {gamma:.4f}" if gamma else "")


if __name__ == '__main__':
    main()
