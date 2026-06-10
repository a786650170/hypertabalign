"""
Pair-classification F1 evaluation on the WDC LSPM cell-to-KB pair subtask.

This script repurposes the labelled pair files prepared for Ditto
(`prep_wdc_lspm_cell_to_kb.py` -> {train,valid,test}.txt) to report
Precision / Recall / F1 in the binary "match / non-match" framing that is
the de-facto WDC Products standard (Peeters & Bizer 2022, Steiner+ 2024).
This makes our cosine bi-encoder baselines directly comparable to
WDC literature, which only reports pair-F1.

Pair file format (one example per line, tab separated):
    <cell-side serialized>\\t<entity-side serialized>\\t<label 0/1>

Both sides are in `[COL] field [VAL] value ...` form. We strip the
COL/VAL markup at load time so backbones never fine-tuned with these
specials (e.g. vanilla DeBERTa) see clean text.

For each model we:
  1. Encode every unique cell-side and entity-side string.
  2. Compute cosine similarity for every pair in valid.txt and pick
     the threshold that maximises F1.
  3. Apply that threshold to test.txt and report Precision/Recall/F1.

Models supported (cosine bi-encoders only — Ditto is a native cross-encoder
classifier and reports its own pair F1 from `ditto_baseline.py`):
  * vanilla       — pre-trained DeBERTa-v3-base, CLS pool, NO fine-tune
  * rsupcon       — `RSupConModel` checkpoint (mean pool, no projection head)
  * biencoder     — `BiEncoderModel` checkpoint (Linear-ReLU-Linear projection)
  * hypertabalign — `HyperGraphRAGModel` checkpoint, KB-encoder branch only
                    (text-only inference path; the GNN branch is bypassed
                    because pair-classification has no row/column context)

Output: appends one row per model to results/pair_f1_table.csv.
"""

import argparse
import os
import re
import sys
import csv

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None


PAIR_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "baselines/ditto_official/data/wdc_lspm_cell_to_kb",
)
DEFAULT_VALID = os.path.join(PAIR_DATA_DIR, "valid.txt")
DEFAULT_TEST  = os.path.join(PAIR_DATA_DIR, "test.txt")


COLVAL_RE = re.compile(r"\[COL\]\s*([^\[]+?)\s*\[VAL\]\s*([^\[]*)")


def strip_colval(serialized):
    parts = COLVAL_RE.findall(serialized)
    if not parts:
        return serialized.strip()
    return " | ".join(f"{f.strip()}: {v.strip()}" for f, v in parts)


def load_pairs(path):
    lefts, rights, labels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            lefts.append(strip_colval(parts[0]))
            rights.append(strip_colval(parts[1]))
            labels.append(int(parts[2]))
    return lefts, rights, labels


def encode_with(encoder_fn, texts, batch_size=64, desc="encoding"):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        emb = encoder_fn(batch)
        out.append(emb.detach().cpu())
    return F.normalize(torch.cat(out, 0), p=2, dim=-1)


def best_threshold(sims, labels):
    best_f1, best_thr = -1.0, 0.0
    sims_np = sims.numpy()
    labels_np = labels.numpy()
    for k in range(-200, 201):
        thr = k / 200.0
        preds = (sims_np >= thr).astype("int64")
        if preds.sum() == 0 and labels_np.sum() > 0:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            labels_np, preds, average="binary", zero_division=0,
        )
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# ---------- Model loaders ----------------------------------------------------

def make_vanilla_encoder(model_name, device):
    from models.encoder.text_backbone import TextEncoder
    enc = TextEncoder(model_name, multi_gpu=False).to(device).eval()
    @torch.no_grad()
    def fn(texts):
        return enc(texts)
    return fn


def make_rsupcon_encoder(model_name, ckpt_path, device):
    from baselines.rsupcon_baseline import RSupConModel
    model = RSupConModel(model_name=model_name).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    @torch.no_grad()
    def fn(texts):
        tk = model.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(device)
        return model.encode(tk["input_ids"], tk["attention_mask"])
    return fn


def make_biencoder_encoder(model_name, ckpt_path, device):
    from baselines.biencoder_train import BiEncoderModel
    model = BiEncoderModel(model_name=model_name).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    @torch.no_grad()
    def fn(texts):
        return model.encode(texts, batch_size=64)
    return fn


def make_hypertabalign_encoder(model_name, ckpt_path, device, batch_size=32):
    """HyperTabAlign in text-only inference mode.

    Pair-classification has no row/column context, so we cannot use the
    table-side GNN branch. Instead we route both sides through the KB
    encoder branch, which is the same shared text backbone followed by
    the model's name-projection layer (the path branch automatically
    disables for single-string inputs). This is the same code path the
    model uses to embed entity names at inference, applied symmetrically
    to cells and entity names — it isolates the text-matching capability
    learned by the joint training.
    """
    from models.unified_model import HyperGraphRAGModel
    cfg = {
        "model_name": model_name,
        "gnn_layers": 2,
        "gnn_hidden_dim": 768,
        "retrieval_dim": 256,
    }
    model = HyperGraphRAGModel(cfg).to(device).eval()

    state = torch.load(ckpt_path, map_location=device)
    # Project's training script wraps state dict under one of {model,
    # model_state_dict, state_dict}; pick whichever holds an actual tensor map.
    if isinstance(state, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    # Strip DDP "module." prefix if present.
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[HyperTabAlign] Loaded ckpt; missing={len(missing)}, "
          f"unexpected={len(unexpected)} (first 3 missing: {missing[:3]})")
    if len(missing) > 50:
        raise RuntimeError(
            f"Too many missing keys ({len(missing)}) — checkpoint structure "
            f"does not match model. Sample missing: {missing[:5]}"
        )

    @torch.no_grad()
    def fn(texts):
        # Empty paths string ⇒ taxonomy branch is skipped (single category root).
        empty_paths = [""] * len(texts)
        return model.kb_encoder(texts, empty_paths)
    return fn


# ---------- Main -------------------------------------------------------------

def evaluate_pair_f1(encoder_fn, valid_path, test_path, batch_size, tag, out_csv):
    print(f"\n==== {tag} ====")
    v_l, v_r, v_lab = load_pairs(valid_path)
    t_l, t_r, t_lab = load_pairs(test_path)
    print(f"  Valid pairs: {len(v_lab)}  (positives: {sum(v_lab)})")
    print(f"  Test  pairs: {len(t_lab)}  (positives: {sum(t_lab)})")

    uniq_left  = list({s for s in v_l + t_l})
    uniq_right = list({s for s in v_r + t_r})
    print(f"  Unique left:  {len(uniq_left)}")
    print(f"  Unique right: {len(uniq_right)}")

    L_emb = encode_with(encoder_fn, uniq_left,  batch_size, "encode-left")
    R_emb = encode_with(encoder_fn, uniq_right, batch_size, "encode-right")
    l_idx = {s: i for i, s in enumerate(uniq_left)}
    r_idx = {s: i for i, s in enumerate(uniq_right)}

    def sims(ls, rs):
        a = L_emb[[l_idx[s] for s in ls]]
        b = R_emb[[r_idx[s] for s in rs]]
        return (a * b).sum(-1)

    v_sims = sims(v_l, v_r)
    t_sims = sims(t_l, t_r)
    v_lab_t = torch.tensor(v_lab)
    t_lab_t = torch.tensor(t_lab)

    thr, v_f1 = best_threshold(v_sims, v_lab_t)
    test_pred = (t_sims.numpy() >= thr).astype("int64")
    p, r, f1, _ = precision_recall_fscore_support(
        t_lab_t.numpy(), test_pred, average="binary", zero_division=0,
    )
    print(f"  Threshold (max-F1 on valid): {thr:+.3f}  (valid F1 = {v_f1:.4f})")
    print(f"  Test  Precision: {p:.4f}")
    print(f"  Test  Recall:    {r:.4f}")
    print(f"  Test  F1:        {f1:.4f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Method", "Threshold", "Valid-F1", "Pair-Precision",
            "Pair-Recall", "Pair-F1",
        ])
        if write_header:
            w.writeheader()
        w.writerow({
            "Method": tag, "Threshold": f"{thr:+.3f}",
            "Valid-F1": f"{v_f1:.4f}",
            "Pair-Precision": f"{p:.4f}",
            "Pair-Recall": f"{r:.4f}",
            "Pair-F1": f"{f1:.4f}",
        })
    print(f"  Appended to {out_csv}")


def main():
    p = argparse.ArgumentParser(description="Pair-F1 on WDC LSPM cell-to-KB subtask")
    p.add_argument("--model", required=True,
                   choices=["vanilla", "rsupcon", "biencoder", "hypertabalign"])
    p.add_argument("--model_name", required=True,
                   help="HF id or local path (e.g. ./models_cache/deberta-v3-base "
                        "or ./models_cache/roberta-base).")
    p.add_argument("--checkpoint", default="",
                   help="Required for rsupcon/biencoder; ignored for vanilla.")
    p.add_argument("--tag", required=True, help="Display name for the results CSV.")
    p.add_argument("--valid", default=DEFAULT_VALID)
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_csv", default=os.path.join(PROJECT_ROOT, "results/pair_f1_table.csv"))
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.model == "vanilla":
        enc_fn = make_vanilla_encoder(args.model_name, device)
    elif args.model == "rsupcon":
        assert args.checkpoint, "--checkpoint required for rsupcon"
        enc_fn = make_rsupcon_encoder(args.model_name, args.checkpoint, device)
    elif args.model == "biencoder":
        assert args.checkpoint, "--checkpoint required for biencoder"
        enc_fn = make_biencoder_encoder(args.model_name, args.checkpoint, device)
    elif args.model == "hypertabalign":
        assert args.checkpoint, "--checkpoint required for hypertabalign"
        enc_fn = make_hypertabalign_encoder(args.model_name, args.checkpoint, device)
    else:
        raise ValueError(args.model)

    evaluate_pair_f1(enc_fn, args.valid, args.test, args.batch_size, args.tag, args.out_csv)


if __name__ == "__main__":
    main()
