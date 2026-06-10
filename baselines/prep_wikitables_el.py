"""
WikiTables-EL data prep:
  in:  tables.json.gz (TabEL/Bhagavatula 2015 corpus, one JSON per line)
  out: data/datasets/wikitables_el/{kb.jsonl, eval/gt.csv, eval/cells.tsv}
       suitable for the vanilla DeBERTa baseline retrieval pipeline.

For every cell that carries at least one Wikipedia surfaceLink we emit:
  - one row in eval/gt.csv:  "wikitables_el,<global_cell_idx>,0,<wiki_title>"
  - one row in eval/cells.tsv: "<global_cell_idx>\t<cell_text>\t<row_context>"
  - the linked Wikipedia title is added to the KB (kb.jsonl: {"id": title, "name": readable_title}).

We split tables 80/10/10 train/val/test deterministically by table _id hash.
Only the test split is needed for the generalisation baseline (matches what
RoCEL/TabEL evaluate on).

This is a STANDALONE prep script. Outputs are consumed by:
  - vanilla_deberta_baseline.py   (drops cells.tsv into eval_utils.load_eval_samples)
  - rsupcon_baseline.py / etc.
"""
import gzip
import hashlib
import json
import os
import sys
from collections import Counter

IN_PATH  = "./data/wikitables_el/tables.json.gz"
OUT_DIR  = "./data/wikitables_el"
OUT_KB   = os.path.join(OUT_DIR, "kb.jsonl")
OUT_GT   = os.path.join(OUT_DIR, "eval/gt.csv")
OUT_CELL = os.path.join(OUT_DIR, "eval/cells.tsv")
OUT_STATS = os.path.join(OUT_DIR, "prep_stats.json")
os.makedirs(os.path.dirname(OUT_GT), exist_ok=True)


def split_of(table_id: str) -> str:
    h = int(hashlib.md5(table_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if h < 8:
        return "train"
    if h == 8:
        return "val"
    return "test"


def title_to_readable(title: str) -> str:
    return title.replace("_", " ")


def main():
    kb_seen = {}   # title -> name
    cells_test = []  # (global_idx, text, row_context, gold_title)
    n_tables, n_cells, n_linked = 0, 0, 0
    split_counter = Counter()

    print(f"[prep] reading {IN_PATH} ...")
    with gzip.open(IN_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = t.get("_id") or str(t.get("pgId", "?"))
            split = split_of(tid)
            split_counter[split] += 1
            n_tables += 1

            data = t.get("tableData", [])
            for ri, row in enumerate(data):
                # Row context = concatenation of all cell texts in the row.
                row_text = " | ".join(
                    (c.get("text") or "").strip()
                    for c in row if (c.get("text") or "").strip()
                )
                for ci, cell in enumerate(row):
                    txt = (cell.get("text") or "").strip()
                    if not txt:
                        continue
                    n_cells += 1
                    links = cell.get("surfaceLinks") or []
                    if not links:
                        continue
                    # Take the first Wikipedia surface link with a title.
                    title = None
                    for L in links:
                        tgt = L.get("target") or {}
                        title = tgt.get("title")
                        if title:
                            break
                    if not title:
                        continue
                    n_linked += 1
                    if title not in kb_seen:
                        kb_seen[title] = title_to_readable(title)
                    if split == "test":
                        global_idx = f"{tid}__{ri}__{ci}"
                        cells_test.append((global_idx, txt, row_text, title))

            if n_tables % 50_000 == 0:
                print(f"  parsed {n_tables} tables, {n_cells} cells, "
                      f"{n_linked} linked, KB size={len(kb_seen)}")

    print(f"[prep] DONE parse: tables={n_tables} cells={n_cells} "
          f"linked={n_linked} KB={len(kb_seen)}")
    print(f"[prep] split distribution: {dict(split_counter)}")
    print(f"[prep] test-set linked cells = {len(cells_test)}")

    # KB jsonl: one entity per line {id, name}.
    print(f"[prep] writing KB to {OUT_KB} ...")
    with open(OUT_KB, "w", encoding="utf-8") as f:
        for title, name in kb_seen.items():
            f.write(json.dumps({"id": title, "name": name}) + "\n")

    # eval/gt.csv & eval/cells.tsv (matches the WDC LSPM eval layout
    # consumed by eval_utils.load_eval_samples).
    print(f"[prep] writing eval split ({len(cells_test)} cells) ...")
    with open(OUT_GT, "w", encoding="utf-8") as g, \
         open(OUT_CELL, "w", encoding="utf-8") as c:
        for global_idx, txt, row, gold in cells_test:
            g.write(f"wikitables_el,{global_idx},0,{gold}\n")
            c.write(f"{global_idx}\t{txt}\t{row}\n")

    stats = {
        "tables": n_tables,
        "cells": n_cells,
        "linked_cells": n_linked,
        "kb_size": len(kb_seen),
        "test_cells": len(cells_test),
        "split_distribution": dict(split_counter),
    }
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[prep] stats -> {OUT_STATS}")


if __name__ == "__main__":
    main()
