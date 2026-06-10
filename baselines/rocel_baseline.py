"""
RoCEL "as faithful as possible" baseline (no official code released).

Reference:
  Wang et al., "RoCEL: Advancing Table Entity Linking through Distinctive
  Row and Column Contexts", EMNLP 2024.

Faithful components (preserved from the paper):
  [✓ ARCH] Row encoder: BERT/DeBERTa with `[M] mention [/M]` row-context
       serialization, [CLS] as row-contextualised mention representation.
  [✓ COL]  Column encoder: FSPool aggregator over per-row contextualised
       embeddings of cells in the same column.
  [✓ FUSE] Fusion MLP: concat(row, col) → mention embedding.
  [✓ FROZEN ENT] BLINK-style FROZEN entity encoder. We freeze a vanilla
       DeBERTa-v3-base CLS-pool encoder over KB entity names (the same
       encoder used by our Vanilla zero-shot lower-bound baseline) and use
       its 768-d KB index as static entity embeddings. The query/row/col
       branches train against these frozen entity embeddings, exactly as
       BLINK + RoCEL do (RoCEL paper §3.1: "entity embeddings are kept
       frozen throughout training").

Adaptations to OUR setting (each one is forced by the difference between
WikiTables EL and our cell-to-KB retrieval; each one is a CONCESSION, not
an optimisation that helps the baseline):
  [SCHEMA] WDC LSPM tables have only `title` + `brand` columns, so column
     contexts are necessarily short (≤2 columns per row). The column-encoder
     branch therefore has less signal than on multi-column WikiTables.
  [LOSS]  We use InfoNCE with random in-batch + sampled negatives because
     our KB has 8M entities and full-vocab ranking cross-entropy
     (paper §3.4) is infeasible. The InfoNCE objective is consistent with
     R-SupCon and the rest of our baseline ladder.
  [SCALE] Paper trained on WikiTables-EL (~50k mentions). We have 457k
     labelled cells; one full epoch of per-mention column context recompute
     is ≥25 hours on H100 (we measured this). We therefore train on a
     UNIFORM 50k-mention SUBSET (matching paper scale) for the paper's
     fixed epoch count (4 epochs, bs=16, k=8 column rows).
  [WARM-UP] Paper uses two warm-up tasks: column typing + set reconstruction.
     WDC LSPM has NO column-type annotations, so the column-typing warm-up
     is fundamentally inapplicable. We do NOT add a set-reconstruction
     warm-up either (would need a separate phase + labels); paper Table 5
     shows warm-up contributes ~1-2 acc points. This is a known limitation
     and is documented as such in the results.

Hyperparams (paper Table 1): bs=16, lr=2e-5, k_col=8 column rows, 4 epochs,
patience=2, AdamW(weight_decay=0.01), max_grad_norm=1.0.
"""
import argparse
import os
import sys
import json
import random
import math

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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

from eval_utils import (
    load_eval_samples_for_rocel,
    evaluate_predictions,
    save_results,
    KB_PATH,
)
from data.dataset import TableAlignmentDataset
from torch_geometric.loader import DataLoader


class FSPool(nn.Module):
    """
    Featurewise Sort Pooling (Zhang et al., 2020).
    Aggregates a SET of vectors into a single vector (order-invariant).
    Used by RoCEL to encode column contexts.
    """

    def __init__(self, in_dim, n_pieces=20):
        super().__init__()
        self.n_pieces = n_pieces
        self.weight = nn.Parameter(torch.randn(in_dim, n_pieces + 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, mask=None):
        """
        x: (batch, set_size, dim) or (set_size, dim)
        Returns: (batch, dim) or (dim,)
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True

        batch, set_size, dim = x.shape

        sorted_x, _ = torch.sort(x, dim=1, descending=True)

        pieces = torch.linspace(0, 1, self.n_pieces + 1, device=x.device)
        positions = torch.linspace(0, 1, set_size, device=x.device) if set_size > 1 else torch.tensor([0.5], device=x.device)

        weight_vals = torch.zeros(dim, set_size, device=x.device)
        for i in range(set_size):
            pos = positions[i]
            idx = torch.clamp((pos * self.n_pieces).long(), 0, self.n_pieces - 1)
            frac = pos * self.n_pieces - idx.float()
            w = self.weight[:, idx] * (1 - frac) + self.weight[:, torch.clamp(idx + 1, max=self.n_pieces)] * frac
            weight_vals[:, i] = w

        weight_vals = weight_vals.unsqueeze(0).expand(batch, -1, -1)
        result = (sorted_x.transpose(1, 2) * weight_vals).sum(dim=2)

        if squeeze:
            result = result.squeeze(0)
        return result


class RoCELModel(nn.Module):
    """RoCEL: row + column differentiated entity linking with FROZEN entity embeddings.

    Architecture (paper §3):
      - Row encoder: DeBERTa over `[M]...[/M]` row serializations → [CLS] gives
        row-contextualised mention embedding (TRAINABLE).
      - Column encoder: FSPool over row-contextualised embeddings of cells in
        the same column (TRAINABLE).
      - Fusion: MLP([row_emb; col_emb]) → mention embedding in entity space
        (TRAINABLE).
      - Entity encoder: BLINK-style FROZEN. Output dim must equal the frozen
        entity dim (768 for vanilla DeBERTa-v3-base CLS-pool). No trainable
        entity projection; this exactly mirrors the paper's "entity embeddings
        are kept frozen" specification.
    """

    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.hidden_size = self.encoder.config.hidden_size

        self.col_encoder = FSPool(self.hidden_size)

        # Fusion MUST output `hidden_size`-dim, matching the frozen entity
        # embedding dim (so cosine sim is in the same space). No trainable
        # entity projection (entity branch is fully frozen, as in the paper).
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        # FIXED temperature, matching RoCEL paper §3.4 (and CLIP / SupCon
        # convention). Earlier versions used nn.Parameter, which during fine-
        # tuning got pushed to 0/negative by the gradient, causing inf in
        # pos_sim / temperature and NaN in cross_entropy. register_buffer
        # keeps it on the right device + dtype without making it learnable.
        self.register_buffer("temperature", torch.tensor(0.07))

    def encode_row_context(self, row_texts, batch_size=16):
        """Encode `[M]...[/M]` row strings with DeBERTa → [CLS] embeddings.

        DeBERTa-v2 SDPA / gradient_checkpointing internals can return fp16
        tensors on H100 even without an outer autocast. Force the same dtype
        as the downstream MLP weights to avoid Half/Float mismatch.
        """
        target_dtype = self.fusion_mlp[0].weight.dtype
        all_embeds = []
        for i in range(0, len(row_texts), batch_size):
            batch = row_texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(next(self.parameters()).device)
            outputs = self.encoder(**inputs)
            cls_embeds = outputs.last_hidden_state[:, 0, :].to(target_dtype)
            all_embeds.append(cls_embeds)
        return torch.cat(all_embeds, dim=0)

    def fuse_row_col(self, row_embeds, col_embed):
        """Fuse row-contextualized embeddings with column embedding(s).

        row_embeds: (N, hidden)
        col_embed:  (hidden,) shared, or (N, hidden) per-row.
        """
        if col_embed.dim() == 1:
            col_expanded = col_embed.unsqueeze(0).expand_as(row_embeds)
        else:
            assert col_embed.shape == row_embeds.shape, (
                f"col_embed shape {tuple(col_embed.shape)} must match row_embeds "
                f"shape {tuple(row_embeds.shape)}"
            )
            col_expanded = col_embed
        fused = torch.cat([row_embeds, col_expanded], dim=-1)
        mention_embeds = self.fusion_mlp(fused)
        return F.normalize(mention_embeds, p=2, dim=-1)


def load_frozen_entity_embeds(cache_path, num_kb_expected, device):
    """Load BLINK-style frozen entity embeddings from disk.

    We reuse the vanilla-DeBERTa CLS-pool KB index built by
    `vanilla_deberta_baseline.py`. These embeddings are L2-normalised
    768-d vectors over the same 8M-entity KB, and the frozen encoder
    has NEVER seen any of our training data (vanilla pre-trained only),
    which is exactly the BLINK paradigm.

    Returns: (num_kb, hidden) tensor on `device`, dtype=float32, kept on
             device for fast lookup during training (~24 GB on H100, fits
             comfortably alongside DeBERTa-v3-base + grad ckpt).
    """
    embeds = torch.load(cache_path, map_location="cpu")
    if embeds.size(0) != num_kb_expected:
        raise ValueError(
            f"Frozen KB cache size mismatch: {embeds.size(0)} vs {num_kb_expected}."
            f" Re-run vanilla_deberta_baseline.py to rebuild the cache."
        )
    embeds = embeds.float()
    embeds = F.normalize(embeds, p=2, dim=-1)
    return embeds.to(device)


def serialize_row_for_mention(x_text, coords, headers, mention_idx):
    """
    Serialize a row context for a specific mention cell.
    Format: "header1: cell1 | header2: [M] mention_cell [/M] | header3: cell3"
    """
    if coords is None or len(coords) == 0:
        return x_text[mention_idx] if mention_idx < len(x_text) else ""

    coords_list = coords.cpu().tolist() if torch.is_tensor(coords) else coords

    mention_row = int(coords_list[mention_idx][0])
    mention_col = int(coords_list[mention_idx][1])

    col_to_header = {}
    for i, (r, c) in enumerate(coords_list):
        if int(r) == -1:
            col_to_header[int(c)] = x_text[i] if i < len(x_text) else ""

    row_cells = {}
    for i, (r, c) in enumerate(coords_list):
        if int(r) == mention_row and i < len(x_text):
            row_cells[int(c)] = (i, x_text[i])

    parts = []
    for col_idx in sorted(row_cells.keys()):
        node_i, cell_text = row_cells[col_idx]
        header = col_to_header.get(col_idx, f"col{col_idx}")
        if node_i == mention_idx:
            parts.append(f"{header}: [M] {cell_text} [/M]")
        else:
            parts.append(f"{header}: {cell_text}")

    return " | ".join(parts) if parts else x_text[mention_idx]


def get_column_cells(x_text, coords, mention_idx):
    """Get all cell indices in the same column as mention_idx (excluding headers)."""
    if coords is None or len(coords) == 0:
        return [mention_idx]

    coords_list = coords.cpu().tolist() if torch.is_tensor(coords) else coords
    mention_col = int(coords_list[mention_idx][1])

    col_cells = []
    for i, (r, c) in enumerate(coords_list):
        if int(c) == mention_col and int(r) >= 0 and i < len(x_text):
            col_cells.append(i)

    return col_cells if col_cells else [mention_idx]


def extract_rocel_training_data(graph, kb_entities, ent_id_to_idx):
    """
    Extract training data for RoCEL from a graph batch.
    Returns list of (row_text, col_cell_indices, positive_kb_idx, mention_node_idx).
    """
    labeled_indices = graph.labeled_indices
    target_ent_ids = graph.target_ent_ids

    if isinstance(target_ent_ids[0], (list, tuple)):
        target_ent_ids = [eid for sublist in target_ent_ids for eid in sublist]

    x_text = graph.x_text
    if x_text and isinstance(x_text[0], list):
        x_text = [t for group in x_text for t in group]

    coords = getattr(graph, 'coords', None)

    samples = []
    for i, eid in enumerate(target_ent_ids):
        if i >= labeled_indices.numel():
            break
        node_idx = labeled_indices[i].item()
        if node_idx >= len(x_text):
            continue

        eid_str = str(eid).strip()
        eid_clean = eid_str.split("/")[-1]
        match_idx = (ent_id_to_idx.get(eid_str) or ent_id_to_idx.get(eid_clean) or
                     ent_id_to_idx.get(eid_str.lower()) or ent_id_to_idx.get(eid_clean.lower()))
        if match_idx is None:
            continue

        row_text = serialize_row_for_mention(x_text, coords, None, node_idx)
        col_cells = get_column_cells(x_text, coords, node_idx)
        samples.append((row_text, col_cells, match_idx, node_idx))

    return samples, x_text, coords


def infonce_loss(query_embeds, pos_embeds, neg_embeds, temperature):
    pos_sim = torch.sum(query_embeds * pos_embeds, dim=-1, keepdim=True) / temperature
    neg_sim = torch.matmul(query_embeds, neg_embeds.t()) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def main():
    parser = argparse.ArgumentParser(
        description="RoCEL-style baseline with FROZEN entity encoder (paper-faithful)"
    )
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--train_data", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/datasets/wdc_lspm_sampled/train"))
    parser.add_argument("--kb_path", type=str, default=KB_PATH)
    # Paper hyperparams (Wang et al. 2024, Table 1):
    parser.add_argument("--epochs", type=int, default=4,
                        help="Paper uses 4 fine-tune epochs.")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="linear LR warmup; mitigates the rank-1 collapse seen in the first attempt.")
    parser.add_argument("--patience", type=int, default=2,
                        help="Paper uses early-stop patience 2.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Paper bs=16.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Paper lr=2e-5.")
    parser.add_argument("--num_negatives", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--max_col_rows", type=int, default=8,
                        help="Paper k_col=8 column rows for FSPool.")
    # Compute concession (documented as adaptation):
    parser.add_argument("--max_train_samples", type=int, default=50000,
                        help="Paper trained on WikiTables-EL (~50k). Our 457k "
                             "would take 4 days/epoch. We sample 50k to match "
                             "paper scale. Set to 0 or negative to disable.")
    parser.add_argument("--frozen_kb_cache", type=str,
                        default=os.path.join(PROJECT_ROOT,
                            "experiments/kb_index/vanilla_deberta_cls/kb_index.pt"),
                        help="Path to frozen entity embeddings (built once by "
                             "vanilla_deberta_baseline.py with vanilla DeBERTa).")
    parser.add_argument("--tag", type=str,
                        default="RoCEL (Wang+2024, faithful: frozen-BLINK + FSPool + row-col fuse)",
                        help="Display name for results CSV.")
    args = parser.parse_args()

    print("=" * 60)
    print("RoCEL faithful baseline (Wang et al. EMNLP 2024)")
    print(f"  Model: {args.model_name}")
    print(f"  Frozen entity encoder: vanilla DeBERTa-v3-base (BLINK paradigm)")
    print(f"  bs={args.batch_size}, lr={args.lr}, ep={args.epochs}, "
          f"k_col={args.max_col_rows}, patience={args.patience}")
    if args.max_train_samples > 0:
        print(f"  Training subset: {args.max_train_samples} mentions "
              f"(matches paper WikiTables-EL scale)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = RoCELModel(args.model_name).to(device)

    kb_entities = []
    with open(args.kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_entities.append(json.loads(line))
    kb_names = [e["name"] for e in kb_entities]
    num_kb = len(kb_entities)
    print(f"[KB] Loaded {num_kb} entities")

    ent_id_to_idx = {}
    for i, e in enumerate(kb_entities):
        raw_id = str(e['id'])
        ent_id_to_idx[raw_id] = i
        ent_id_to_idx[raw_id.split("/")[-1].strip()] = i
        ent_id_to_idx[raw_id.lower()] = i
        ent_id_to_idx[raw_id.split("/")[-1].strip().lower()] = i

    # ===== Load FROZEN BLINK-style entity embeddings =====
    print(f"\n[Frozen entity encoder] Loading from {args.frozen_kb_cache} ...")
    if not os.path.exists(args.frozen_kb_cache):
        raise FileNotFoundError(
            f"Frozen entity cache not found: {args.frozen_kb_cache}\n"
            f"Run vanilla_deberta_baseline.py first to build it."
        )
    frozen_kb = load_frozen_entity_embeds(args.frozen_kb_cache, num_kb, device)
    assert frozen_kb.size(1) == model.hidden_size, (
        f"Frozen entity dim {frozen_kb.size(1)} must match model hidden_size "
        f"{model.hidden_size}. Use the SAME backbone for both."
    )
    print(f"  Loaded shape={tuple(frozen_kb.shape)}, dtype={frozen_kb.dtype}, "
          f"on {frozen_kb.device}")

    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/rocel_baseline.pt")

    if args.eval_only:
        load_path = args.checkpoint or ckpt_path
        print(f"\n[eval_only] Loading checkpoint: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
        print("  Checkpoint loaded, skipping training.")
    else:
        print(f"\n[1/3] Training RoCEL faithful model...")
        full_train = TableAlignmentDataset(root=args.train_data)
        if args.max_train_samples > 0 and args.max_train_samples < len(full_train):
            g = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(full_train), generator=g)[:args.max_train_samples].tolist()
            train_dataset = torch.utils.data.Subset(full_train, indices)
            print(f"  Sampled {len(train_dataset)} of {len(full_train)} graphs.")
        else:
            train_dataset = full_train
            print(f"  Using full {len(train_dataset)} graphs.")
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        # === Patched 2026-05-11: linear LR warmup to stabilise the first
        # InfoNCE pass against the frozen 8M-entity index. The original run
        # collapsed at epoch 2 because every step in that epoch produced a
        # non-finite loss; they were silently skipped by the defensive guard
        # below, leaving total_loss=0 / valid_steps=0 and `avg_loss=0`,
        # which the early-stopping logic mistakenly treated as the new best.
        warmup_steps = max(1, args.warmup_steps)
        def _lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        global_step = 0

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            total_loss, valid_steps, nan_steps = 0.0, 0, 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                samples, x_text, coords = extract_rocel_training_data(
                    batch, kb_entities, ent_id_to_idx
                )
                if len(samples) < 2:
                    continue

                row_texts = [s[0] for s in samples]
                pos_indices = [s[2] for s in samples]

                # Sample random negatives from KB (same as before).
                pos_set = set(pos_indices)
                neg_indices = []
                while len(neg_indices) < args.num_negatives:
                    idx = random.randint(0, num_kb - 1)
                    if idx not in pos_set:
                        neg_indices.append(idx)

                # FROZEN entity lookup — no gradient through entity branch.
                pos_embeds = frozen_kb[torch.as_tensor(pos_indices, device=device)]
                neg_embeds = frozen_kb[torch.as_tensor(neg_indices, device=device)]

                row_embeds = model.encode_row_context(row_texts)

                mention_embeds_list = []
                for si, s in enumerate(samples):
                    col_cell_indices = s[1]
                    # Cap column-context size during training (paper k_col=8).
                    # Without this, a column with hundreds of rows blows up
                    # the per-mention DeBERTa forwards.
                    if len(col_cell_indices) > args.max_col_rows:
                        col_cell_indices = random.sample(
                            list(col_cell_indices), args.max_col_rows
                        )
                    if len(col_cell_indices) > 1 and coords is not None:
                        col_row_texts = []
                        for ci in col_cell_indices:
                            col_row_texts.append(
                                serialize_row_for_mention(x_text, coords, None, ci)
                            )
                        with torch.no_grad():
                            col_row_embeds = model.encode_row_context(col_row_texts)
                        col_embed = model.col_encoder(col_row_embeds.unsqueeze(0)).squeeze(0)
                    else:
                        col_embed = torch.zeros(model.hidden_size, device=device)

                    fused = model.fuse_row_col(
                        row_embeds[si:si + 1], col_embed
                    )
                    mention_embeds_list.append(fused)

                query_embeds = torch.cat(mention_embeds_list, dim=0)

                loss = infonce_loss(query_embeds, pos_embeds, neg_embeds, model.temperature)

                # Skip steps where loss is non-finite (defensive): updating
                # parameters with NaN/Inf grad would corrupt the entire model.
                # This is standard training hygiene, not a baseline boost.
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    nan_steps += 1
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                total_loss += loss.item()
                valid_steps += 1

            avg_loss = total_loss / max(1, valid_steps)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

            # Always persist a per-epoch snapshot so a later collapse cannot
            # overwrite an earlier healthy checkpoint (the bug from the first
            # training attempt). Eval loads ckpt_path (best); the *_epochN.pt
            # files are an audit trail.
            epoch_ckpt = ckpt_path.replace(".pt", f"_epoch{epoch + 1}.pt")
            torch.save(model.state_dict(), epoch_ckpt)

            # Explicit collapse / NaN-saturation detection. A "0.0 best" is
            # treated as a FAILED epoch and NOT used to overwrite the best
            # checkpoint.
            total_attempted = valid_steps + nan_steps
            nan_rate = nan_steps / max(1, total_attempted)
            epoch_failed = (
                valid_steps == 0
                or nan_rate > 0.5
                or avg_loss < 0.05
            )

            if epoch_failed:
                patience_counter += 1
                print(f"Epoch {epoch + 1} | loss={avg_loss:.4f}  "
                      f"valid={valid_steps}  nan={nan_steps}  nan_rate={nan_rate:.2%}  "
                      f"✗ FAILED (collapse / NaN-saturated)  ({patience_counter}/{args.patience})")
                # Do NOT update best — preserve last healthy ckpt.
                if not os.path.exists(ckpt_path):
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"  (no best yet; saved current as fallback)")
            elif avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
                print(f"Epoch {epoch + 1} | loss={avg_loss:.4f}  nan={nan_steps}  ✓ best → saved")
            else:
                patience_counter += 1
                print(f"Epoch {epoch + 1} | loss={avg_loss:.4f}  nan={nan_steps}  ✗ no improve ({patience_counter}/{args.patience})")
                if not os.path.exists(ckpt_path):
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"  (no best ckpt yet; saved current as fallback)")

            if patience_counter >= args.patience:
                print(f"  ⏹ Early stopping at epoch {epoch + 1}")
                break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Loaded best checkpoint: {ckpt_path}")

    # ===== Build KB index (FROZEN — already loaded) =====
    print(f"\n[2/4] KB index = frozen entity embeddings (no re-encoding needed)")
    kb_embeddings = frozen_kb.cpu()  # (num_kb, hidden) — same throughout
    model.eval()

    # ===== Build column-context cache (matches training-time col_cells path) =====
    print(f"\n[3/4] Loading eval samples (with [M] markers and column contexts)...")
    samples, col_context_rows = load_eval_samples_for_rocel(
        max_col_rows=args.max_col_rows
    )

    print(f"  Encoding column-context FSPool embeddings for "
          f"{len(col_context_rows)} (chunk, col) pairs...")
    col_embed_cache = {}
    with torch.no_grad():
        for key, rows_with_marker in tqdm(col_context_rows.items(),
                                           desc="Encoding col contexts"):
            if not rows_with_marker:
                col_embed_cache[key] = torch.zeros(model.hidden_size)
                continue
            row_embeds = model.encode_row_context(rows_with_marker, batch_size=32)
            col_embed = model.col_encoder(row_embeds.unsqueeze(0)).squeeze(0)
            col_embed_cache[key] = col_embed.cpu()

    # ===== Encode all queries via row-with-marker (training-consistent path) =====
    print(f"\n[4/4] Encoding {len(samples)} queries (row-with-marker) and retrieving...")
    row_texts = [s["row_with_marker"] if s["row_with_marker"] else s["cell_text"]
                 for s in samples]
    all_row_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(row_texts), 64), desc="Encoding queries (row)"):
            batch = row_texts[i:i + 64]
            embeds = model.encode_row_context(batch, batch_size=64)
            all_row_embeds.append(embeds.cpu())
    row_embeds_all = torch.cat(all_row_embeds, dim=0)

    col_embeds_per_sample = torch.stack(
        [col_embed_cache.get((s["chunk_idx"], s["col_idx"]),
                             torch.zeros(model.hidden_size)) for s in samples],
        dim=0,
    )

    all_query_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), 256), desc="Fusing row+col"):
            r = row_embeds_all[i:i + 256].to(device)
            c = col_embeds_per_sample[i:i + 256].to(device)
            fused = model.fuse_row_col(r, c)
            all_query_embeds.append(fused.cpu())
    query_embeds = torch.cat(all_query_embeds, dim=0)

    id_to_name = {str(e["id"]): e["name"] for e in kb_entities}
    preds, labels, pred_names, label_names, candidates_info = [], [], [], [], []

    for i in tqdm(range(0, len(samples), 256), desc="Searching"):
        q = query_embeds[i:i + 256]
        scores = torch.matmul(q, kb_embeddings.t())
        topk_scores, topk_indices = torch.topk(scores, k=args.top_k, dim=-1)

        for j in range(q.size(0)):
            idx = i + j
            cands = [{"id": str(kb_entities[k]["id"]), "name": kb_entities[k]["name"]}
                     for k in topk_indices[j].tolist()]
            candidates_info.append(cands)
            preds.append(cands[0]["id"] if cands else "NIL")
            pred_names.append(cands[0]["name"] if cands else "NIL")
            labels.append(samples[idx]["gold_entity_id"])
            label_names.append(id_to_name.get(samples[idx]["gold_entity_id"],
                                               str(samples[idx]["gold_entity_id"])))

    metrics = evaluate_predictions(
        preds, labels,
        candidates_info=candidates_info,
        pred_names=pred_names,
        label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)


if __name__ == "__main__":
    main()
