# HyperTabAlign

Official code release for the paper:

**HyperTabAlign: Disentangled Hypergraph Encoding and Multi-Task Contrastive Retrieval for Million-Scale Table Entity Alignment**

Yufei Tang¹², Chunmian Wang¹, Yi Tian¹², Xinyu Lin¹², Chengbin Hou¹ (corresponding author)

¹ Fuyao University of Science and Technology, China
² Xiamen University, China

Contact: `2025801003@stu.fyust.edu.cn` (first author), `chengbin.hou10@foxmail.com` (corresponding author)

---

## Overview

HyperTabAlign is a structure-aware retrieval framework for aligning web-table cells to a million-scale knowledge base. It factorizes the table into two independent hyperedge views — row-wise attribute complementarity and column-wise type consistency — propagates each through a dedicated attentive `HypergraphConv` branch, fuses them with a learnable per-node gate, and trains a shared 256-dimensional retrieval head with multi-positive InfoNCE, live hard-negative mining, name-space consistency, and an auxiliary entity-recognition head.

On the **WDC LSPM** benchmark (8M candidate entities, 114k labeled eval cells), HyperTabAlign attains **0.8438 top-1 Accuracy** in the recommended *direct* deployment mode, exceeding the strongest supervised contrastive baseline (R-SupCon) by 2.3 points and the strongest zero-shot retriever (MPNet) by 1.7 points.

Key empirical finding: structural supervision contributes most strongly **during training**, not at inference. Under matched conditions, removing the hypergraph branches during training costs roughly 2.9 Accuracy points, while removing them only at inference time has negligible impact. Practitioners can therefore ship HyperTabAlign-direct as a standard sub-millisecond bi-encoder while preserving the structural gains collected at training time.

---

## Repository structure

```
hypertabalign/
├── models/
│   ├── unified_model.py           # top-level wrapper that ties everything together
│   ├── encoder/
│   │   ├── table_gnn.py           # GATv2 + dual HypergraphConv + per-node gate
│   │   └── text_backbone.py       # DeBERTa-v3-base wrapper, mean / CLS pooling
│   ├── knowledge/
│   │   ├── kb_encoder.py          # name + optional taxonomy-path projection
│   │   ├── manager.py             # 8M-entity FP32 index management
│   │   └── normalizer.py          # string normalization for Name-Accuracy
│   ├── retrieval/dual_encoder.py  # shared 256-d retrieval head with learnable τ
│   └── alignment/scorer.py        # cosine-similarity scorer
├── training/
│   ├── trainer.py                 # multi-task joint training loop, live hard-neg mining
│   └── losses.py                  # multi-positive InfoNCE, name-space consistency
├── data/
│   └── dataset.py                 # TableAlignmentDataset over Parquet shards
├── baselines/
│   ├── rsupcon_baseline.py        # R-SupCon faithful re-implementation
│   ├── ditto_baseline.py          # Ditto cross-encoder + 200k pair pool
│   ├── ablation_runner.py         # row_only / col_only / no_hgnn / no_gnn variants
│   ├── pair_f1_eval.py            # WDC pair-classification F1 protocol
│   ├── vanilla_deberta_baseline.py
│   ├── eval_utils.py              # shared Acc / MRR@10 / Hit@K / Name-Acc routines
│   └── llm_eval.py                # ComEM / pairwise / TailorMatch wrappers
├── scripts/
│   ├── train.py                   # main training entry point
│   ├── measure_efficiency.py      # cold-build + warm-query latency
│   ├── qual_dump_*.py             # per-cell top-1 dumps for qualitative tables
│   ├── merge_qual_cases.py        # build the qualitative case table (Table VII)
│   ├── gate_dump.py               # per-cell α_v + γ extraction (Fig. 5)
│   ├── scale_curve_*.py           # Accuracy vs KB-size sweep (Fig. 6)
│   ├── plot_*.py                  # paper-ready figure rendering
│   ├── mpnet_zero_shot_eval.py    # MPNet (all-mpnet-base-v2) baseline
│   └── vanilla_deberta_zero_shot_eval.py
├── configs/base_config.yaml
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/a786650170/hypertabalign.git
cd hypertabalign
pip install -r requirements.txt
```

The code is tested on Python 3.10–3.12 with PyTorch 2.x and torch-geometric ≥ 2.5.

---

## Data and Checkpoints

**WDC LSPM** is publicly available from `https://webdatacommons.org/largescaleproductcorpus/lspm/`. We use a sampled cell-to-KB-entity alignment construction; the exact sampled split (train / valid / eval gt.csv + targets.csv + Parquet shards) and the 8M-entity KB JSONL will be released alongside the trained `wdc_sota_v2.pt` checkpoint via the project release page once the paper is camera-ready.

Expected directory layout once data is in place:

```
./data/wdc_lspm_sampled/
    ├── train/{targets.csv,gt.csv,tables/chunks/*.parquet}
    ├── eval/{targets.csv,gt.csv,tables/chunks/*.parquet}
    └── wdc_products_kb.jsonl       # 8,043,290 entities
./checkpoints/
    └── wdc_sota_v2.pt              # released after camera-ready
```

---

## Reproducing the main paper results

| Paper artifact | Command |
|---|---|
| Table I row "HyperTabAlign (direct)" | `python scripts/qual_dump_hypertab.py --ckpt ./checkpoints/wdc_sota_v2.pt` |
| Table II ablations | `python baselines/ablation_runner.py --mode {no_hgnn,row_only,col_only,no_contrastive,...}` |
| Table III direct vs hypergraph | `python run.py --eval_only --ckpt ./checkpoints/wdc_sota_v2.pt --mode {hyper,direct}` |
| Table IV efficiency | `python scripts/measure_efficiency.py --ckpt ./checkpoints/wdc_sota_v2.pt` |
| Table V pair-classification F1 | `python baselines/pair_f1_eval.py --model {vanilla,rsupcon,mpnet,hypertabalign}` |
| Table VII qualitative cases | `python scripts/qual_dump_hypertab.py && python scripts/qual_dump_rsupcon.py && python scripts/qual_dump_text.py --model mpnet && python scripts/merge_qual_cases.py` |
| Fig. 3 loss curves | `python scripts/plot_loss_curves.py` |
| Fig. 4 similarity distributions | `python scripts/plot_sim_distribution.py` |
| Fig. 5 gate weight distribution | `python scripts/gate_dump.py && python scripts/plot_gate_distribution.py` |
| Fig. 6 Accuracy vs KB size | `python scripts/scale_curve_hypertab.py && python scripts/scale_curve_rsupcon.py && python scripts/plot_scale_curve.py` |

---

## Training from scratch

```bash
# 4×H100 distributed data parallel, BF16, per-rank batch size 196
torchrun --nproc_per_node=4 run.py \
    --train_dir ./data/wdc_lspm_sampled/train \
    --kb_path   ./data/wdc_lspm_sampled/wdc_products_kb.jsonl \
    --num_train_negatives 96 \
    --hard_negative_topk  48 \
    --hard_negative_random_ratio 0.25 \
    --save_path ./checkpoints/wdc_sota_v2.pt
```

Training proceeds until validation loss stops improving. The best-validation checkpoint is what we report in the paper.

---

## Inference

```python
import torch
from models.unified_model import HyperGraphRAGModel

config = dict(model_name="microsoft/deberta-v3-base",
              gnn_layers=2, gnn_hidden_dim=768, retrieval_dim=256)
model = HyperGraphRAGModel(config).cuda().eval()
state = torch.load("./checkpoints/wdc_sota_v2.pt", map_location="cuda", weights_only=False)
model.load_state_dict(state["model"], strict=False)

# direct mode (recommended for deployment): bypass the GNN entirely
# embed every cell text through the shared retrieval head
e = model(graph=None, mode="encode_kb", names=[cell_text], paths=[""])
q = torch.nn.functional.normalize(model.retriever.shared_proj(e), p=2, dim=-1)
# search against the pre-built KB index ...
```

The KB index is built once with the same shared projection (see `models/knowledge/manager.py`); at 8M entities in 256-d FP32 it occupies 8.2 GB on a single GPU and serves chunked top-K queries at sub-millisecond latency.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tang2027hypertabalign,
  title     = {HyperTabAlign: Disentangled Hypergraph Encoding and Multi-Task Contrastive
               Retrieval for Million-Scale Table Entity Alignment},
  author    = {Tang, Yufei and Wang, Chunmian and Tian, Yi and Lin, Xinyu and Hou, Chengbin},
  booktitle = {Proceedings of the IEEE International Conference on Data Engineering (ICDE)},
  year      = {2027}
}
```

---

## License

MIT License. See `LICENSE`.
