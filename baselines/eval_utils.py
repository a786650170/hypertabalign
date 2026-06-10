"""
Shared utilities for all baseline experiments.
Loads eval data (cell texts + gold labels) and provides unified evaluation.
"""
import os
import sys
import json
import csv
import random
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import Evaluator
from models.knowledge.normalizer import normalize_entity_id

EVAL_DIR = os.path.join(PROJECT_ROOT, "data/datasets/wdc_lspm_sampled/eval")
CHUNKS_DIR = os.path.join(EVAL_DIR, "tables/chunks")
KB_PATH = os.path.join(PROJECT_ROOT, "data/datasets/wdc_lspm/wdc_products_kb.jsonl")
PHYSICAL_CHUNK_SIZE = 10000


def load_eval_samples(eval_dir=EVAL_DIR, chunks_dir=CHUNKS_DIR):
    """Load eval data: returns list of dicts with cell_text, row_context, gold_entity_id."""
    gt_path = os.path.join(eval_dir, "gt.csv")
    samples = []
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            table_id, abs_row, col, ent_id = row[0], int(row[1]), int(row[2]), row[3]
            ent_id = str(ent_id).split("/")[-1]
            samples.append({
                "table_id": table_id,
                "abs_row": abs_row,
                "col": col,
                "gold_entity_id": ent_id,
            })

    chunk_cache = {}
    for s in samples:
        abs_row = s["abs_row"]
        chunk_idx = abs_row // PHYSICAL_CHUNK_SIZE
        row_in_chunk = abs_row % PHYSICAL_CHUNK_SIZE

        if chunk_idx not in chunk_cache:
            pq = os.path.join(chunks_dir, f"wdc_lspm_part_{chunk_idx}.parquet")
            csv_path = os.path.join(chunks_dir, f"wdc_lspm_part_{chunk_idx}.csv")
            if os.path.exists(pq):
                chunk_cache[chunk_idx] = pd.read_parquet(pq, engine="pyarrow")
            elif os.path.exists(csv_path):
                chunk_cache[chunk_idx] = pd.read_csv(csv_path)
            else:
                chunk_cache[chunk_idx] = None

        df = chunk_cache[chunk_idx]
        if df is not None and row_in_chunk < len(df):
            cols = df.columns.tolist()
            cell_text = str(df.iloc[row_in_chunk, s["col"]]) if s["col"] < len(cols) else ""
            row_vals = [str(df.iloc[row_in_chunk, c]) for c in range(len(cols))]
            row_context = " | ".join(f"{cols[c]}: {row_vals[c]}" for c in range(len(cols)))
        else:
            cell_text = ""
            row_context = ""

        s["cell_text"] = cell_text
        s["row_context"] = row_context

    print(f"[eval_utils] Loaded {len(samples)} eval samples.")
    return samples


def _serialize_row_with_marker(df_row, headers, target_col):
    """Serialize a DataFrame row with [M]...[/M] markers around the target column.
    Format: "header1: cell1 | header2: [M] cell [/M] | header3: cell3"
    """
    parts = []
    n = len(headers)
    for c in range(n):
        cell = str(df_row.iloc[c]) if c < len(df_row) else ""
        header = str(headers[c])
        if c == target_col:
            parts.append(f"{header}: [M] {cell} [/M]")
        else:
            parts.append(f"{header}: {cell}")
    return " | ".join(parts)


def load_eval_samples_for_rocel(eval_dir=EVAL_DIR, chunks_dir=CHUNKS_DIR,
                                max_col_rows=32, seed=42):
    """RoCEL-specific eval loader: returns (samples, col_context_rows).

    Each sample contains (in addition to fields from load_eval_samples):
      - row_with_marker: "header1: cell1 | header2: [M] cell [/M] | ..." matching
        the training-time serialization in rocel_baseline.serialize_row_for_mention.
      - chunk_idx, row_in_chunk, col_idx: keys for column-context cache.

    col_context_rows: dict[(chunk_idx, col_idx)] -> list[str]
        For each (chunk_idx, col_idx) pair, up to max_col_rows row_with_marker
        strings (each marking the cell of that column in a different row).
        Used to build per-column FSPool embeddings during RoCEL evaluation.

    The marker format and per-row serialization match the training pipeline
    (rocel_baseline.serialize_row_for_mention), eliminating the train/eval
    distribution mismatch that disabled the row encoder + col branch in earlier
    versions.
    """
    rng = random.Random(seed)

    gt_path = os.path.join(eval_dir, "gt.csv")
    samples = []
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            table_id, abs_row, col, ent_id = row[0], int(row[1]), int(row[2]), row[3]
            ent_id = str(ent_id).split("/")[-1]
            samples.append({
                "table_id": table_id,
                "abs_row": abs_row,
                "col": col,
                "gold_entity_id": ent_id,
            })

    chunk_cache = {}

    def _get_chunk(chunk_idx):
        if chunk_idx in chunk_cache:
            return chunk_cache[chunk_idx]
        pq = os.path.join(chunks_dir, f"wdc_lspm_part_{chunk_idx}.parquet")
        csv_path = os.path.join(chunks_dir, f"wdc_lspm_part_{chunk_idx}.csv")
        if os.path.exists(pq):
            df = pd.read_parquet(pq, engine="pyarrow")
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = None
        chunk_cache[chunk_idx] = df
        return df

    for s in samples:
        chunk_idx = s["abs_row"] // PHYSICAL_CHUNK_SIZE
        row_in_chunk = s["abs_row"] % PHYSICAL_CHUNK_SIZE
        s["chunk_idx"] = chunk_idx
        s["row_in_chunk"] = row_in_chunk
        s["col_idx"] = s["col"]

        df = _get_chunk(chunk_idx)
        if df is not None and row_in_chunk < len(df) and s["col"] < len(df.columns):
            cols = df.columns.tolist()
            row = df.iloc[row_in_chunk]
            cell_text = str(row.iloc[s["col"]])
            row_vals = [str(row.iloc[c]) for c in range(len(cols))]
            row_context = " | ".join(f"{cols[c]}: {row_vals[c]}" for c in range(len(cols)))
            row_with_marker = _serialize_row_with_marker(row, cols, s["col"])
        else:
            cell_text = ""
            row_context = ""
            row_with_marker = ""

        s["cell_text"] = cell_text
        s["row_context"] = row_context
        s["row_with_marker"] = row_with_marker

    print(f"[eval_utils] Loaded {len(samples)} RoCEL eval samples.")

    unique_keys = {(s["chunk_idx"], s["col_idx"]) for s in samples if s["row_with_marker"]}
    col_context_rows = {}
    for chunk_idx, col_idx in unique_keys:
        df = _get_chunk(chunk_idx)
        if df is None or col_idx >= len(df.columns):
            col_context_rows[(chunk_idx, col_idx)] = []
            continue
        cols = df.columns.tolist()
        n_rows = len(df)
        if n_rows > max_col_rows:
            row_ids = sorted(rng.sample(range(n_rows), max_col_rows))
        else:
            row_ids = list(range(n_rows))
        rows_with_marker = [
            _serialize_row_with_marker(df.iloc[ri], cols, col_idx) for ri in row_ids
        ]
        col_context_rows[(chunk_idx, col_idx)] = rows_with_marker

    print(f"[eval_utils] Built col-context for {len(col_context_rows)} "
          f"(chunk_idx, col_idx) pairs (max_col_rows={max_col_rows}).")
    return samples, col_context_rows


def load_kb(kb_path=KB_PATH, max_entities=None):
    """Load KB entities as list of dicts with id, name, path."""
    entities = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_entities and i >= max_entities:
                break
            entities.append(json.loads(line))
    print(f"[eval_utils] Loaded {len(entities)} KB entities from {kb_path}")
    return entities


def evaluate_predictions(preds, labels, candidates_info=None,
                         pred_names=None, label_names=None, tag="Baseline"):
    """Run the project's Evaluator and print results."""
    evaluator = Evaluator()
    metrics = evaluator.compute(
        preds, labels,
        candidates_info=candidates_info,
        pred_names=pred_names,
        label_names=label_names,
        debug=False,
    )
    print(f"\n[{tag}] {len(preds)} samples")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def save_results(metrics, tag, results_dir=None):
    """Append metrics to results CSV."""
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "comparison_table.csv")

    row = {"Method": tag}
    row.update({k: f"{v:.4f}" for k, v in metrics.items()})

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[eval_utils] Results saved to {out_path}")
