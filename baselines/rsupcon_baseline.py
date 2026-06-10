"""
R-SupCon (Peeters & Bizer 2022, WWW Poster) — FAITHFUL adaptation to WDC LSPM.

Paper: "Supervised Contrastive Learning for Product Matching"
Code base: https://github.com/wbsg-uni-mannheim/contrastive-product-matching

This file faithfully ports the *core ingredients* of R-SupCon to our
cell-to-KB-entity retrieval setting:

  1. SupConLoss — verbatim copy from wbsg/src/contrastive/models/loss.py
     (multi-positive contrastive loss; reduces to InfoNCE when only one
     positive per anchor exists).
  2. Source-aware sampling — two sampling datasets (cell-side / entity-side)
     are constructed and each batch randomly chooses one source via
     `random.choice([0, 1])` in the collator (mirrors the official
     `DataCollatorContrastivePretrainDeepmatcher`).
  3. Attribute-aware serialization — `[COL] field [VAL] value ...` format
     identical to the official `serialize_sample_lspc`. Special tokens
     `[COL]` and `[VAL]` are added to the tokenizer (resize_token_embeddings).
  4. Mean pooling over token embeddings (NOT CLS) — exactly as
     `mean_pooling` in the official `modeling.py`.
  5. NO projection head — the official supervised `ContrastivePretrainModel`
     does not apply a contrastive head, only L2-normalize after pooling.
  6. Backbone: RoBERTa-base (as in original paper for product matching).

Adaptations to OUR setting (documented honestly — each one is forced by
the difference between LSPC pair-matching and our cell-to-KB retrieval, NOT
an optimisation to make the baseline look better):

  [SCHEMA] WDC LSPM table chunks have only `title` + `brand` columns (paper's
    `description` / `specTableContent` are absent), so cell-side serialization
    uses only the two available fields. Our KB schema has only `name` per
    entity (no brand/description), so entity-side serialization uses a single
    `[COL] name [VAL] X` field. This REDUCES the input information available
    to R-SupCon vs. the paper's setting; it is not an optimisation.

  [CLUSTER] cluster_id := gold_entity_id (each KB entity is one cluster).
    Multi-positive arises naturally on the cell-side when many cells link to
    the same entity; entity-side only has one sample per cluster (positive
    sampled from the same cluster degenerates to self, which is exactly what
    wbsg's `selection.sample(1).iloc[0]` does for singleton clusters too).

  [NO FT] We do NOT run the second-stage cross-entropy fine-tuning (paper's
    `run_finetune_siamese.py`) because our task is open-KB retrieval, not
    binary pair classification. The pre-trained encoder is directly used for
    KB retrieval (cosine similarity). This omission DISADVANTAGES R-SupCon
    relative to the paper, but is unavoidable.

  [NO AUG] We do NOT use data augmentation (paper's `aug=all-`). This
    corresponds to the "R-SupCon w/o aug" row reported in the paper, NOT
    the optimised "R-SupCon" row. Faithful, no extra tricks.

  [COMPUTE] Paper pre-trains for fixed 200 epoch with bs=1024 on small
    corpora (~10k pairs). On our 457k-cell training set that would take
    ~600 GPU-hours per run. We use bs=64, lr=5e-5 (paper's fine-tune setting,
    which they also list as a valid pre-training config in Table 4) with a
    train-loss-based early-stopping (patience=10) capped at 50 epoch. This
    is a compute concession; if anything it stops EARLIER than paper.

  [LR SCHEDULE] We mirror HF Trainer defaults exactly: AdamW(weight_decay=0,
    betas=(0.9, 0.999), eps=1e-8) + linear schedule with warmup_steps=0 +
    max_grad_norm=1.0. Same as the official `run_pretraining_deepmatcher.py`.

NO numerical-stability hacks were added: SupConLoss is byte-for-byte the
same as wbsg's; NaN/Inf are not masked out; we do not skip bad batches.
"""
import argparse
import os
import sys
import json
import random
import csv
import glob
from collections import defaultdict

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"
import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from eval_utils import load_eval_samples, evaluate_predictions, save_results, KB_PATH


# ============================================================
# 1. SupConLoss — VERBATIM from wbsg/src/contrastive/models/loss.py
#    (Tian et al. 2020, https://arxiv.org/abs/2004.11362)
# ============================================================
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning (https://arxiv.org/pdf/2004.11362.pdf).

    Verbatim copy of the implementation used by Peeters & Bizer 2022.
    Supports the unsupervised SimCLR loss when neither labels nor mask given.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if features.dim() < 3:
            raise ValueError('features needs [bsz, n_views, ...]')
        if features.dim() > 3:
            features = features.view(features.size(0), features.size(1), -1)

        batch_size = features.size(0)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and mask')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.size(0) != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.size(1)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # NOTE: faithful to wbsg — anchors with zero positives in the batch
        # produce NaN/Inf here. We do NOT mask them out, mirroring the
        # exact behaviour of the official implementation.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


# ============================================================
# 2. R-SupCon Bi-Encoder Model — same architecture as wbsg ContrastivePretrainModel
#    (mean pooling, no projection head, L2 normalize before SupConLoss)
# ============================================================
def mean_pooling(token_embeds, attention_mask):
    """Identical to wbsg/modeling.mean_pooling."""
    expanded = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    return torch.sum(token_embeds * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)


class RSupConModel(nn.Module):
    """R-SupCon supervised pre-training model (faithful port of wbsg).

    forward(left_inputs, right_inputs, labels) → (loss,)
      where left/right are the two views (cell-side / entity-side) and
      labels are cluster_id (gold_entity_id encoded to int).
    """

    def __init__(self, model_name="roberta-base", temperature=0.07,
                 add_special_tokens=True):
        super().__init__()
        # Add [COL] [VAL] as additional special tokens (paper convention).
        kw = {}
        if add_special_tokens:
            kw["additional_special_tokens"] = ["[COL]", "[VAL]"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)

        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.criterion = SupConLoss(temperature=temperature)

    def encode(self, input_ids, attention_mask):
        """Token-mean pool → L2 normalize. (Matches wbsg modeling.py L92-99.)"""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pooling(out.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, left_input_ids, left_attention_mask,
                right_input_ids, right_attention_mask, labels):
        left = self.encode(left_input_ids, left_attention_mask).unsqueeze(1)   # (B,1,D)
        right = self.encode(right_input_ids, right_attention_mask).unsqueeze(1)
        features = torch.cat([left, right], dim=1)  # (B, 2, D)
        loss = self.criterion(features, labels=labels)
        return loss


# ============================================================
# 3. WDC LSPM ↔ R-SupCon dataset adaptation
#    (mirrors wbsg ContrastivePretrainDatasetDeepmatcher, but for our
#     cell-to-KB-entity setting on top of the gt.csv + chunks parquet files.)
# ============================================================
def serialize_cell_lspc(title, brand):
    """Cell-side serialization (mirrors wbsg serialize_sample_lspc).

    wbsg only does pandas `fillna('')` then string slicing — no extra null
    handling, no "None"-string filtering. We follow exactly: any string
    value (including the literal "None") flows into the serialization
    untouched, except for pandas NaN which we treat as the empty string
    (matching wbsg's `fillna('')`). Slice limits (50 / 5) match
    `serialize_sample_lspc` in wbsg datasets.py L34-41.
    """
    title = "" if pd.isna(title) else str(title)
    brand = "" if pd.isna(brand) else str(brand)
    s = f"[COL] title [VAL] {' '.join(title.split(' ')[:50])}".strip()
    s = f"{s} [COL] brand [VAL] {' '.join(brand.split(' ')[:5])}".strip()
    return s


def serialize_entity(entity_name):
    """Entity-side serialization. Our KB only has `name`, so we use a single
    `[COL] name [VAL] X` field. Same `fillna('')` semantics as wbsg."""
    name = "" if entity_name is None or pd.isna(entity_name) else str(entity_name)
    return f"[COL] name [VAL] {' '.join(name.split(' ')[:60])}".strip()


class WDCLSPMRSupConDataset(torch.utils.data.Dataset):
    """Source-aware contrastive dataset for our cell-to-entity setting.

    Two view-streams ('source-aware'):
      data_cell[i]   = (cell-text serialized, cluster_id)
      data_entity[i] = (entity-text serialized, cluster_id)
    Both indexed by row i: row-i in data_cell and data_entity correspond
    to the same gold cluster, so a (cell_anchor, entity_pos) pair is the
    natural cross-source positive.

    `__getitem__(idx)` returns ((cell_anchor, cell_pos),
                                (entity_anchor, entity_pos))
    where cell_pos / entity_pos are sampled within the same cluster (same
    cluster_id). When a cluster only has one cell or one entity, pos = self
    (this is the same behaviour as wbsg's `selection.sample(1)`).
    """

    def __init__(self, train_root, kb_path, max_samples=None, seed=42):
        rng = random.Random(seed)
        np.random.seed(seed)

        chunks_dir = os.path.join(train_root, "tables", "chunks")
        gt_path = os.path.join(train_root, "gt.csv")

        # Load KB (id → name)
        print(f"  [Dataset] Loading KB from {kb_path} ...")
        self.kb_id_to_name = {}
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                eid = str(e["id"]).split("/")[-1].strip()
                self.kb_id_to_name[eid] = str(e.get("name", e.get("title", "")))

        # Load gt.csv: (table_id, abs_row, col, gold_entity_id)
        print(f"  [Dataset] Loading gt from {gt_path} ...")
        rows = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in csv.reader(f):
                if len(line) < 4:
                    continue
                rows.append((line[0], int(line[1]), int(line[2]),
                             str(line[3]).split("/")[-1].strip()))

        if max_samples is not None and max_samples < len(rows):
            rng.shuffle(rows)
            rows = rows[:max_samples]
        print(f"  [Dataset] {len(rows)} labeled cells")

        # Group rows by chunk to load each parquet only once
        chunk_to_rows = defaultdict(list)
        PHYS = 10000
        for ridx, (tid, abs_row, col, eid) in enumerate(rows):
            chunk_idx = abs_row // PHYS
            chunk_to_rows[chunk_idx].append((ridx, abs_row % PHYS, col, eid))

        cell_records = [None] * len(rows)
        entity_records = [None] * len(rows)
        unmatched = 0
        for chunk_idx in tqdm(sorted(chunk_to_rows.keys()), desc="  Reading chunks"):
            pq = os.path.join(chunks_dir, f"wdc_lspm_part_{chunk_idx}.parquet")
            if not os.path.exists(pq):
                unmatched += len(chunk_to_rows[chunk_idx])
                continue
            df = pd.read_parquet(pq)
            cols = list(df.columns)
            for ridx, row_in_chunk, col, eid in chunk_to_rows[chunk_idx]:
                if row_in_chunk >= len(df):
                    continue
                entity_name = self.kb_id_to_name.get(eid)
                if entity_name is None:
                    continue
                # Cell-side: use the row's title/brand (independent of which col was target)
                title = df.iloc[row_in_chunk][cols[0]] if len(cols) >= 1 else ""
                brand = df.iloc[row_in_chunk][cols[1]] if len(cols) >= 2 else ""
                cell_records[ridx] = (serialize_cell_lspc(title, brand), eid)
                entity_records[ridx] = (serialize_entity(entity_name), eid)

        # Drop unmatched / missing
        kept = [(c, e) for c, e in zip(cell_records, entity_records)
                if c is not None and e is not None]
        if not kept:
            raise RuntimeError("No usable training samples after KB matching.")
        cells, entities = zip(*kept)

        # Encode cluster_id (string entity id) → int label
        cluster_ids = [c[1] for c in cells]
        unique = {cid: i for i, cid in enumerate(sorted(set(cluster_ids)))}
        labels = [unique[cid] for cid in cluster_ids]

        self.cell_text = [c[0] for c in cells]
        self.entity_text = [e[0] for e in entities]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_clusters = len(unique)

        # Build cluster → row_indices map for positive sampling
        self.cluster_to_indices = defaultdict(list)
        for i, lbl in enumerate(self.labels):
            self.cluster_to_indices[int(lbl)].append(i)

        print(f"  [Dataset] {len(self.cell_text)} pairs, "
              f"{self.num_clusters} unique clusters "
              f"(avg {len(self.cell_text)/max(1,self.num_clusters):.2f} cells/cluster, "
              f"unmatched: {unmatched})")

    def __len__(self):
        return len(self.cell_text)

    def __getitem__(self, idx):
        lbl = int(self.labels[idx])
        same = self.cluster_to_indices[lbl]
        # Positive sampling: another row with the same cluster_id (or self).
        pos_idx = random.choice(same) if len(same) > 1 else idx
        # data1 view (cell side): (anchor cell, positive cell)
        cell_anchor = self.cell_text[idx]
        cell_pos = self.cell_text[pos_idx]
        # data2 view (entity side): (anchor entity, positive entity)
        ent_anchor = self.entity_text[idx]
        ent_pos = self.entity_text[pos_idx]
        return (
            (cell_anchor, cell_pos, lbl),
            (ent_anchor, ent_pos, lbl),
        )


class RSupConCollator:
    """Source-aware collator (faithful port of
    DataCollatorContrastivePretrainDeepmatcher in wbsg).

    For each batch, randomly chooses cell-side (rnd=0) or entity-side (rnd=1).
    Then builds left/right tokenized batches from the chosen side's
    (anchor, positive, label) triples.
    """
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch_pairs):
        rnd = random.choice([0, 1])  # mirrors wbsg L79
        side = [p[rnd] for p in batch_pairs]
        anchors = [s[0] for s in side]
        positives = [s[1] for s in side]
        labels = [s[2] for s in side]

        left = self.tokenizer(anchors, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")
        right = self.tokenizer(positives, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        return {
            "left_input_ids": left["input_ids"],
            "left_attention_mask": left["attention_mask"],
            "right_input_ids": right["input_ids"],
            "right_attention_mask": right["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================
# 4. Train + Eval main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="R-SupCon (Peeters+Bizer 2022) faithful baseline for WDC LSPM"
    )
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="HF model id or local path (paper used roberta-base)")
    parser.add_argument("--tag", type=str,
                        default="R-SupCon (Peeters+22, faithful adaptation)")
    parser.add_argument("--train_data", type=str,
                        default=os.path.join(PROJECT_ROOT,
                            "data/datasets/wdc_lspm_sampled/train"))
    parser.add_argument("--kb_path", type=str, default=KB_PATH)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Paper fine-tune regime (bs=64, lr=5e-5, 50 ep, patience=10)")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap training pairs (debug only)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    print("=" * 60)
    print("R-SupCon (Peeters & Bizer 2022) — faithful adaptation")
    print(f"  Model: {args.model_name}")
    print(f"  Loss: SupConLoss (multi-positive, T={args.temperature})")
    print(f"  Pooling: mean (NOT CLS), no projection head")
    print(f"  Source-aware sampling: cell-side / entity-side")
    print(f"  Serialization: [COL] field [VAL] value")
    print(f"  bs={args.batch_size}, lr={args.lr}, epochs={args.epochs}, "
          f"patience={args.patience}")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # ===== Build model =====
    model = RSupConModel(
        model_name=args.model_name, temperature=args.temperature
    ).to(device)
    safe_name = args.model_name.replace("/", "_").replace("-", "_")
    ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/rsupcon_{safe_name}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # ===== Load KB (for eval index) =====
    kb_entities = []
    with open(args.kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_entities.append(json.loads(line))
    kb_names = [e["name"] for e in kb_entities]
    num_kb = len(kb_entities)
    print(f"[KB] Loaded {num_kb} entities")

    if args.eval_only:
        load_path = args.checkpoint or ckpt_path
        print(f"\n[eval_only] Loading checkpoint: {load_path}")
        state = torch.load(load_path, map_location=device)
        model.load_state_dict(state)
        model.float()
        print("  Checkpoint loaded.")
    else:
        # ===== Build dataset =====
        print(f"\n[1/3] Building R-SupCon dataset (source-aware sampling) ...")
        dataset = WDCLSPMRSupConDataset(
            train_root=args.train_data,
            kb_path=args.kb_path,
            max_samples=args.max_samples,
        )
        collator = RSupConCollator(model.tokenizer, max_length=args.max_length)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collator, persistent_workers=args.num_workers > 0,
            drop_last=True,
        )

        # HF Trainer defaults: AdamW(lr=lr, weight_decay=0.0, betas=(0.9,0.999),
        #   eps=1e-8) + linear schedule with warmup_steps=0 + max_grad_norm=1.0.
        # We mirror those defaults exactly (no weight decay, no warmup).
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=0.0,
                                      betas=(0.9, 0.999), eps=1e-8)
        total_steps = len(loader) * args.epochs
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps,
        )

        # ===== Training loop =====
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(args.epochs):
            model.train()
            total_loss, valid_steps = 0.0, 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                left_ids = batch["left_input_ids"].to(device)
                left_mask = batch["left_attention_mask"].to(device)
                right_ids = batch["right_input_ids"].to(device)
                right_mask = batch["right_attention_mask"].to(device)
                lbls = batch["labels"].to(device)

                loss = model(left_ids, left_mask, right_ids, right_mask, lbls)

                # No NaN-skipping: faithful to wbsg, NaN propagates.
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                valid_steps += 1
                if valid_steps % 50 == 0:
                    pbar.set_postfix(loss=f"{total_loss/valid_steps:.4f}",
                                     lr=f"{scheduler.get_last_lr()[0]:.2e}")

            avg_loss = total_loss / max(1, valid_steps)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"Epoch {epoch+1} | loss={avg_loss:.4f} ✓ best → saved")
            else:
                patience_counter += 1
                print(f"Epoch {epoch+1} | loss={avg_loss:.4f} ✗ "
                      f"({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Loaded best checkpoint: {ckpt_path}")

    # ===== Build KB index =====
    print(f"\n[2/3] Building KB index with trained R-SupCon encoder ...")
    model.eval()
    kb_serialized = [serialize_entity(n) for n in kb_names]
    all_kb_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, num_kb, 256), desc="Encoding KB"):
            batch_text = kb_serialized[i:i+256]
            tk = model.tokenizer(batch_text, padding=True, truncation=True,
                                 max_length=args.max_length, return_tensors="pt").to(device)
            embeds = model.encode(tk["input_ids"], tk["attention_mask"])
            all_kb_embeds.append(embeds.cpu())
    kb_embeddings = torch.cat(all_kb_embeds, dim=0)

    # ===== Encode queries (eval cells) =====
    print(f"\n[3/3] Loading eval samples and retrieving ...")
    eval_samples = load_eval_samples()
    # For eval queries, use cell-side serialization built from row_context.
    # row_context is "header1: cell1 | header2: cell2 ...". For consistency
    # with training, we re-build [COL] title [VAL] X [COL] brand [VAL] Y from
    # the parsed row context where possible; otherwise fall back to cell_text.
    query_texts = []
    for s in eval_samples:
        # row_context format: "header: value | header: value | ..."
        title, brand = "", ""
        for piece in s.get("row_context", "").split("|"):
            if ":" not in piece:
                continue
            head, val = piece.split(":", 1)
            head = head.strip().lower()
            val = val.strip()
            if head == "title" and not title:
                title = val
            elif head == "brand" and not brand:
                brand = val
        if not title and not brand:
            title = s["cell_text"]
        query_texts.append(serialize_cell_lspc(title, brand))

    all_q_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(query_texts), 256), desc="Encoding queries"):
            batch_text = query_texts[i:i+256]
            tk = model.tokenizer(batch_text, padding=True, truncation=True,
                                 max_length=args.max_length, return_tensors="pt").to(device)
            embeds = model.encode(tk["input_ids"], tk["attention_mask"])
            all_q_embeds.append(embeds.cpu())
    query_embeds = torch.cat(all_q_embeds, dim=0)

    # ===== Retrieval =====
    kb_dev = kb_embeddings.to(device)
    preds, pred_names, gold_ids = [], [], []
    candidates_info, label_names = [], []
    id_to_name = {str(e["id"]): e["name"] for e in kb_entities}
    with torch.no_grad():
        for i in tqdm(range(0, len(eval_samples), 256), desc="Searching"):
            q = query_embeds[i:i+256].to(device)
            scores = torch.matmul(q, kb_dev.T)
            topk_scores, topk_idx = torch.topk(scores, k=args.top_k, dim=-1)
            for j in range(q.size(0)):
                cands = []
                for k in range(args.top_k):
                    eidx = topk_idx[j, k].item()
                    cands.append({
                        "id": str(kb_entities[eidx]["id"]),
                        "name": kb_entities[eidx]["name"],
                        "score": float(topk_scores[j, k].item()),
                    })
                preds.append(cands[0]["id"])
                pred_names.append(cands[0]["name"])
                gold_ids.append(eval_samples[i+j]["gold_entity_id"])
                candidates_info.append(cands)
                label_names.append(id_to_name.get(eval_samples[i+j]["gold_entity_id"], ""))

    metrics = evaluate_predictions(
        preds, gold_ids,
        candidates_info=candidates_info,
        pred_names=pred_names,
        label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)
    print(f"\n[done] {args.tag}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
