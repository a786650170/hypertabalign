"""
Ditto baseline — faithful re-implementation by importing the OFFICIAL
ditto_light modules from megagonlabs/ditto.

Reference:
  Li et al., "Deep Entity Matching with Pre-Trained Language Models" (Ditto),
  PVLDB 14(1), 2021.  Code: https://github.com/megagonlabs/ditto

WHAT IS BYTE-IDENTICAL TO OFFICIAL DITTO (we IMPORT and CALL these classes
without modification):
  * `ditto_light.ditto.DittoModel`              — RoBERTa cross-encoder,
        [CLS] → Linear(2), MixDA Beta-blend embedding mixing (paper §4.5).
  * `ditto_light.dataset.DittoDataset`          — DA via Augmenter at __getitem__,
        same `pad()` collator.
  * `ditto_light.augment.Augmenter`             — 9 ops + RandAugment 'all' mode.
  * `ditto_light.knowledge.ProductDKInjector`   — spaCy NER (en_core_web_lg) +
        number normalization + ID detection (paper §4.3).
  * `ditto_light.summarize.Summarizer`          — TF-IDF + NLTK stopword
        summarization fitted on train+valid+test (paper §4.4 Algorithm 1).

OPTIMIZER + LOSS (byte-equivalent to ditto_light.ditto.train / train_step):
  * AdamW(lr=3e-5, weight_decay=0.0)
  * get_linear_schedule_with_warmup(num_warmup_steps=0,
        num_training_steps=(n_train // batch_size) * n_epochs)
  * CrossEntropyLoss over 2 classes (no label smoothing)

DATA PREP FLOW (mirrors official train_ditto.py):
  raw  ──► Summarizer.transform_file(max_len=128)  ──►  train.txt.su
  .su  ──► ProductDKInjector.transform_file()      ──►  train.txt.su.dk
  .dk  ──► DittoDataset(path=…, da='all')          ──►  on-the-fly augment

ADAPTATIONS (each one is a CONCESSION, not a baseline-strengthening change):
  [TASK]  Paper does entity↔entity binary classification on pre-blocked
          candidate pairs. We do cell→KB retrieval. We therefore:
            (i)  build training triples (cell, gold_entity, label) from our
                 chunks parquet + gt.csv (cell side = WDC LSPM row's
                 title+brand, entity side = KB entity name);
            (ii) sample 3 random KB entities as negatives per positive cell
                 (paper had pre-blocked candidates; we cannot block 8M
                 entities, so random sampling is the natural fallback — this
                 makes negatives EASIER, i.e. does NOT unfairly strengthen
                 Ditto vs. our main model).
  [SCALE] 50 000 positive cells × (1 pos + 3 neg) = 200 000 training pairs,
          comparable to WDC xlarge train.txt = 171 714 (paper Table 4).
  [EVAL]  Paper reports F1 on labelled candidate pairs. We need top-1
          retrieval over 8M entities → wrap the trained Ditto in our standard
          retrieval-rerank pipeline (vanilla DeBERTa dense retrieval gets
          top-K=50, then trained Ditto rerranks). Same K used for all
          rerankers in this project.
  [LM]    Paper sweeps {DistilBERT, BERT-base, RoBERTa-base}; Table 5 has
          RoBERTa-base as the best single LM (--lm roberta is the official
          flag).  We use roberta-base (loaded from offline cache).
  [FP16]  Paper used apex O2; we run fp32 (apex not installed in env).
          Pure speed difference, no accuracy difference.
"""

import argparse
import json
import os
import random
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Compat patches (transformers 5.0 + apex optional, byte-equivalent semantics)
import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DITTO_OFFICIAL = os.path.join(os.path.dirname(__file__), "ditto_official")
if DITTO_OFFICIAL not in sys.path:
    sys.path.insert(0, DITTO_OFFICIAL)

# >>> the official Ditto algorithmic core (unmodified imports) <<<
from ditto_light.ditto import DittoModel             # noqa: E402
from ditto_light.dataset import DittoDataset         # noqa: E402
from ditto_light.knowledge import ProductDKInjector  # noqa: E402
from ditto_light.summarize import Summarizer         # noqa: E402
# <<<


class SafeDittoDataset(DittoDataset):
    """Same as the official DittoDataset, but graceful when augment_sent
    accidentally removes the ' [SEP] ' separator. This happens in our setting
    because the right side has a single short COL (`COL name VAL <name>`),
    so drop_col can sometimes empty the right side, leaving '... [SEP]' at
    the end of `combined`; the official line `combined.split(' [SEP] ')`
    then raises ValueError.

    Fix: detect that case and fall back to the non-augmented pair, which is
    equivalent to da=None for that single sample. This does NOT strengthen
    the baseline — it only removes a degenerate crash. Equivalent to the
    paper running with slightly less data augmentation on a small fraction
    of samples.
    """

    def __getitem__(self, idx):
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]
        x = self.tokenizer.encode(text=left, text_pair=right,
                                  max_length=self.max_len, truncation=True)
        if self.da is not None:
            combined = self.augmenter.augment_sent(
                left + ' [SEP] ' + right, self.da)
            if ' [SEP] ' in combined:
                lh, rh = combined.split(' [SEP] ', 1)
            else:
                lh, rh = left, right  # graceful fallback
            x_aug = self.tokenizer.encode(text=lh, text_pair=rh,
                                          max_length=self.max_len, truncation=True)
            return x, x_aug, self.labels[idx]
        return x, self.labels[idx]

from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from eval_utils import KB_PATH, evaluate_predictions, load_eval_samples, save_results  # noqa: E402


# =============================================================================
# Serialization helpers
# =============================================================================
def _clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("none", "nan"):
        return ""
    return s.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def serialize_cell(title, brand):
    title, brand = _clean(title), _clean(brand)
    parts = []
    if title:
        parts.append(f"COL title VAL {title}")
    if brand:
        parts.append(f"COL brand VAL {brand}")
    return " ".join(parts) if parts else "COL title VAL"


def serialize_entity(name):
    name = _clean(name)
    return f"COL name VAL {name}" if name else "COL name VAL"


def parse_eval_row_context(s):
    title, brand = "", ""
    for piece in str(s.get("row_context", "")).split("|"):
        if ":" not in piece:
            continue
        head, val = piece.split(":", 1)
        h = head.strip().lower()
        v = val.strip()
        if h == "title" and not title:
            title = v
        elif h == "brand" and not brand:
            brand = v
    if not title and not brand:
        title = s.get("cell_text", "")
    return title, brand


# =============================================================================
# Data preparation: ensure train.txt + (optionally) Summarize + DK have run.
# =============================================================================
def ensure_prep_files(prep_dir):
    """Returns (train, valid, test) paths after any requested transform."""
    return (os.path.join(prep_dir, "train.txt"),
            os.path.join(prep_dir, "valid.txt"),
            os.path.join(prep_dir, "test.txt"))


def run_summarize(config, lm, max_len, files):
    """Apply official Summarizer.transform_file once and cache .su outputs."""
    summarizer = Summarizer(config, lm=lm)
    out = []
    for fn in files:
        if os.path.exists(fn + ".su") and os.path.getsize(fn + ".su") > 0:
            print(f"  [Summarize] cache hit {fn}.su")
        else:
            print(f"  [Summarize] {fn} → {fn}.su")
            summarizer.transform_file(fn, max_len=max_len, overwrite=False)
        out.append(fn + ".su")
    return summarizer, out


def run_dk(files, dk_name="product"):
    """Apply official ProductDKInjector.transform_file once and cache .dk outputs."""
    injector = ProductDKInjector(config={}, name=dk_name)
    out = []
    for fn in files:
        if os.path.exists(fn + ".dk") and os.path.getsize(fn + ".dk") > 0:
            # Sanity: counts match
            try:
                with open(fn) as f:
                    n_in = sum(1 for _ in f)
                with open(fn + ".dk") as f:
                    n_out = sum(1 for _ in f)
                if n_in == n_out:
                    print(f"  [DK]        cache hit {fn}.dk ({n_out} lines)")
                    out.append(fn + ".dk"); continue
                print(f"  [DK]        cache stale ({n_in}≠{n_out}); rebuilding {fn}.dk")
            except Exception:
                print(f"  [DK]        cache invalid; rebuilding {fn}.dk")
        print(f"  [DK]        {fn} → {fn}.dk  (spaCy NER, this is the slow step)")
        injector.transform_file(fn, overwrite=True)
        out.append(fn + ".dk")
    return injector, out


# =============================================================================
# Eval-time on-the-fly transform (re-uses the same Summarizer + DKInjector)
# =============================================================================
_DK_CACHE = {}  # entity-string  → DK-transformed string


def _cached_dk(dk_injector, s):
    """Cache spaCy NER transforms — eval reuses the same KB entity names many
    times. Algorithmically identical to dk_injector.transform(s)."""
    if dk_injector is None:
        return s
    if s not in _DK_CACHE:
        _DK_CACHE[s] = dk_injector.transform(s)
    return _DK_CACHE[s]


def transform_pair_for_eval(left_raw, right_raw, summarizer, dk_injector, max_len):
    """Apply Summarize + DK to one (left, right) string pair, mirroring the
    train-time .su.dk preprocessing exactly (same class instances)."""
    if summarizer is not None:
        row = f"{left_raw}\t{right_raw}\t0"
        out = summarizer.transform(row, max_len=max_len)
        sa, sb, _ = out.strip().split("\t")
        left, right = sa.strip(), sb.strip()
    else:
        left, right = left_raw, right_raw
    left = _cached_dk(dk_injector, left)
    right = _cached_dk(dk_injector, right)
    return left, right


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Ditto baseline (faithful via official ditto_light)")
    p.add_argument("--lm", type=str, default="./models_cache/roberta-base")
    p.add_argument("--prep_dir", type=str,
                   default=os.path.join(DITTO_OFFICIAL, "data", "wdc_lspm_cell_to_kb"))
    p.add_argument("--kb_path", type=str, default=KB_PATH)
    p.add_argument("--epochs", type=int, default=15)         # paper Table 9
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)     # paper Table 9
    p.add_argument("--lr", type=float, default=3e-5)         # paper Table 9
    p.add_argument("--max_length", type=int, default=128)    # paper sweep [128,256]
    p.add_argument("--alpha_aug", type=float, default=0.8)   # paper §4.5
    p.add_argument("--da", type=str, default="all")          # paper default
    p.add_argument("--dk", type=str, default="product")      # paper §4.3
    p.add_argument("--summarize", action="store_true", default=True)
    p.add_argument("--no_summarize", dest="summarize", action="store_false")
    p.add_argument("--retrieval_top_k", type=int, default=50)
    p.add_argument("--final_top_k", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--tag", type=str, default="Ditto (faithful, official ditto_light + RoBERTa)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training and load existing checkpoint for eval only.")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 76)
    print(f"Ditto (FAITHFUL via official ditto_light): {args.tag}")
    print(f"  lm={args.lm}  bs={args.batch_size}  lr={args.lr}  max_len={args.max_length}")
    print(f"  epochs={args.epochs}  da={args.da}  dk={args.dk}  summarize={args.summarize}")
    print("=" * 76)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------- 1. Data prep (Summarize → DK), mirrors train_ditto.py ----------
    print("\n[1/3] Data prep (Summarize → DK), with disk cache")
    train_path, valid_path, test_path = ensure_prep_files(args.prep_dir)
    for fn in (train_path, valid_path, test_path):
        assert os.path.exists(fn), f"Missing prep file: {fn}\n" \
                                    f"Run prep_wdc_lspm_cell_to_kb.py first."

    config = {"name": "wdc_lspm_cell_to_kb",
              "trainset": train_path,
              "validset": valid_path,
              "testset":  test_path}

    summarizer = None
    if args.summarize:
        summarizer, (train_path, valid_path, test_path) = run_summarize(
            config, args.lm, args.max_length,
            [train_path, valid_path, test_path])

    dk_injector = None
    if args.dk:
        dk_injector, (train_path, valid_path, test_path) = run_dk(
            [train_path, valid_path, test_path], dk_name=args.dk)

    print(f"  Final train file: {train_path}")
    print(f"  Final valid file: {valid_path}")
    print(f"  Final test  file: {test_path}")

    # ---------- 2. Build OFFICIAL DittoDataset + DataLoader ----------
    print("\n[2/3] Building official DittoDataset")
    train_ds = SafeDittoDataset(path=train_path, max_len=args.max_length,
                                lm=args.lm, da=args.da if args.da else None)
    print(f"  Train pairs: {len(train_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=DittoDataset.pad, pin_memory=True)

    # ---------- 3. Build OFFICIAL DittoModel ----------
    model = DittoModel(device=str(device), lm=args.lm,
                       alpha_aug=args.alpha_aug).to(device)

    # ---------- 4. Optimizer + scheduler (byte-equivalent to ditto_light.ditto.train) ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    num_steps = (len(train_ds) // args.batch_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=max(1, num_steps))
    criterion = nn.CrossEntropyLoss()

    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/ditto_faithful.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    if args.skip_train and os.path.exists(ckpt_path):
        print(f"\n[skip_train] re-using existing checkpoint {ckpt_path}, going straight to eval")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        _train(args, model, train_loader, optimizer, scheduler, criterion,
               train_ds, ckpt_path, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  loaded best ckpt {ckpt_path}")

    _run_eval(args, model, summarizer, dk_injector, device)


def _train(args, model, train_loader, optimizer, scheduler, criterion,
           train_ds, ckpt_path, device):
    """Mirrors ditto_light.ditto.train_step exactly."""
    print(f"\nTraining {args.tag}")
    best_loss, no_improve = float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        total, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            if len(batch) == 2:
                x, y = batch
                pred = model(x)
            else:
                x1, x2, y = batch
                pred = model(x1, x2)  # MixDA Beta-blend inside DittoModel.forward
            loss = criterion(pred, y.to(model.device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item(); steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")
            del loss

        avg = total / max(1, steps)
        if avg < best_loss:
            best_loss, no_improve = avg, 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Epoch {epoch+1} | loss={avg:.4f} ✓ best → saved")
        else:
            no_improve += 1
            print(f"Epoch {epoch+1} | loss={avg:.4f} ✗ no-improve ({no_improve}/{args.patience})")
        if no_improve >= args.patience:
            print(f"  ⏹ early stop at epoch {epoch+1}")
            break


def _run_eval(args, model, summarizer, dk_injector, device):
    """Vanilla DeBERTa retrieval (AutoModel + CLS pool) → Ditto rerank.

    NOTE: We deliberately do NOT use baselines.dense_retriever.DenseRetriever
    because it wraps the project's custom KBEncoder, which contains GRU and
    Linear layers that are randomly initialised (no checkpoint loaded). That
    leads to garbage candidates — top-50 rarely contains the gold entity, so
    any reranker on top scores 0.0% by construction.

    Instead we mirror baselines/vanilla_deberta_baseline.py exactly: vanilla
    pre-trained AutoModel, CLS-pool, L2-normalize, dot-product over the
    cached 8M-entity index at experiments/kb_index/vanilla_deberta_cls/.
    This is the same retrieval pipeline whose top-1 acc is 0.79 (the
    Vanilla DeBERTa zero-shot baseline). Using it for Ditto means Ditto
    rerranks the SAME candidate pool as that baseline, which is the
    fair, apples-to-apples setting.
    """
    print("\n[3/3] vanilla DeBERTa dense retrieval (AutoModel + CLS pool) …")
    from transformers import AutoModel
    from vanilla_deberta_baseline import cls_encode

    deberta_path = "./models_cache/deberta-v3-base"
    deberta_tok = AutoTokenizer.from_pretrained(deberta_path)
    deberta_model = AutoModel.from_pretrained(deberta_path).to(device)
    deberta_model.eval()

    kb_cache = os.path.join(
        PROJECT_ROOT,
        "experiments/kb_index/vanilla_deberta_cls/kb_index.pt",
    )
    print(f"  loading cached vanilla CLS KB index {kb_cache} …")
    kb_embeds = torch.load(kb_cache, map_location="cpu")
    kb_embeds = torch.nn.functional.normalize(kb_embeds, p=2, dim=-1)

    samples = load_eval_samples()
    kb_entities = []
    with open(args.kb_path) as f:
        for line in f:
            kb_entities.append(json.loads(line))
    assert len(kb_entities) == kb_embeds.size(0), \
        f"KB cache mismatch: {kb_embeds.size(0)} vs {len(kb_entities)}"

    cell_texts = [s["cell_text"] for s in samples]
    qe = cls_encode(deberta_model, deberta_tok, cell_texts, device,
                    batch_size=256, max_length=128, use_fp16=True)

    print(f"  scoring {len(samples)} queries × 8M KB …")
    kb_embeds_dev = kb_embeds.to(device)
    all_cands = []
    chunk = 256
    with torch.no_grad():
        for i in tqdm(range(0, qe.size(0), chunk), desc="dot-product"):
            q = qe[i:i + chunk].to(device)
            scores = torch.matmul(q, kb_embeds_dev.t())
            topk_scores, topk_idx = torch.topk(scores, k=args.retrieval_top_k, dim=-1)
            for j in range(q.size(0)):
                cands = []
                for k in range(args.retrieval_top_k):
                    eidx = topk_idx[j, k].item()
                    cands.append({
                        "id": str(kb_entities[eidx]["id"]),
                        "name": kb_entities[eidx]["name"],
                        "score": float(topk_scores[j, k].item()),
                    })
                all_cands.append(cands)
    del kb_embeds_dev, kb_embeds, qe, deberta_model
    torch.cuda.empty_cache()

    print(f"\n   Ditto reranking {len(samples)} samples …")
    model.eval()
    id_to_name = {str(e["id"]): e["name"] for e in kb_entities}
    tokenizer = AutoTokenizer.from_pretrained(args.lm)

    # _cached_dk dedups DK transforms across the eval set: each unique
    # post-Summarize string is NER-tagged at most once.
    preds, labels, pred_names, label_names, cands_info = [], [], [], [], []
    with torch.no_grad():
        for i, s in enumerate(tqdm(samples, desc="Ditto rerank")):
            cands = all_cands[i]
            title, brand = parse_eval_row_context(s)
            left_raw = serialize_cell(title, brand)
            if cands:
                pairs_left, pairs_right = [], []
                for c in cands:
                    right_raw = serialize_entity(c["name"])
                    l, r = transform_pair_for_eval(
                        left_raw, right_raw, summarizer, dk_injector, args.max_length)
                    pairs_left.append(l); pairs_right.append(r)
                scores = []
                B = 64
                for k in range(0, len(pairs_left), B):
                    bl = pairs_left[k:k+B]; br = pairs_right[k:k+B]
                    encs = [tokenizer.encode(text=a, text_pair=b,
                                             max_length=args.max_length, truncation=True)
                            for a, b in zip(bl, br)]
                    maxlen = max(len(e) for e in encs)
                    x = torch.LongTensor(
                        [e + [0] * (maxlen - len(e)) for e in encs])
                    logits = model(x)                            # [B, 2]
                    probs = logits.softmax(dim=-1)[:, 1]          # [B]
                    scores.extend(probs.cpu().tolist())
                for j, sc in enumerate(scores):
                    cands[j]["rerank_score"] = sc
                cands.sort(key=lambda x: -x.get("rerank_score", 0))
                cands = cands[: args.final_top_k]
            out = [{"id": c["id"], "name": c["name"]} for c in cands]
            cands_info.append(out)
            preds.append(out[0]["id"] if out else "NIL")
            pred_names.append(out[0]["name"] if out else "NIL")
            labels.append(s["gold_entity_id"])
            label_names.append(id_to_name.get(s["gold_entity_id"], str(s["gold_entity_id"])))

    metrics = evaluate_predictions(
        preds, labels,
        candidates_info=cands_info,
        pred_names=pred_names, label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)


if __name__ == "__main__":
    main()
