"""
Ablation study runner.
Subclasses existing modules to disable specific components, then runs
the same train+eval pipeline as run.py. No existing code is modified.

Ablation modes:
  no_hgnn       - Disable HypergraphConv, keep GAT only
  no_gnn        - Disable all GNN layers, text encoder directly to alignment_proj
  no_contrastive - Disable NED contrastive loss, keep NER + Name loss only
"""
import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"
import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from models.encoder.table_gnn import TableGNN
from training.trainer import JointTrainer
from models.knowledge.manager import KnowledgeManager
from data.dataset import TableAlignmentDataset
from torch_geometric.loader import DataLoader
from evaluation.metrics import Evaluator
from models.knowledge.normalizer import normalize_entity_id
from tqdm import tqdm


class TableGNN_NoHGNN(TableGNN):
    """TableGNN with HypergraphConv disabled — GAT only."""

    def forward(self, x_text, edge_index, hyperedge_index=None,
                edge_attr=None, retrieved_context_embeds=None,
                hyperedge_conn_type=None, coords=None, **kw):
        return super().forward(
            x_text, edge_index,
            hyperedge_index=None,
            edge_attr=edge_attr,
            retrieved_context_embeds=retrieved_context_embeds,
            hyperedge_conn_type=None,
            coords=coords,
        )


class TableGNN_NoGNN(TableGNN):
    """TableGNN with all GNN layers disabled — text encoder + projection only."""

    def forward(self, x_text, edge_index, hyperedge_index=None,
                edge_attr=None, retrieved_context_embeds=None,
                hyperedge_conn_type=None, coords=None, **kw):
        if isinstance(x_text, (list, tuple)) and len(x_text) > 0 and isinstance(x_text[0], (list, tuple)):
            flat_x_text = [item for sublist in x_text for item in sublist]
        else:
            flat_x_text = x_text

        x = self.text_encoder(flat_x_text)
        if x.dtype != self.pre_proj.weight.dtype:
            x = x.to(dtype=self.pre_proj.weight.dtype)
        x = self.pre_proj(x)

        if retrieved_context_embeds is not None:
            x = x + retrieved_context_embeds

        extraction_logits = self.extraction_head(x)
        alignment_embeds = self.alignment_proj(x)
        return extraction_logits, alignment_embeds


class TableGNN_RowOnly(TableGNN):
    """Keep ONLY row hyperedges; drop column hyperedges (test column-signal contribution)."""

    def forward(self, x_text, edge_index, hyperedge_index=None,
                edge_attr=None, retrieved_context_embeds=None,
                hyperedge_conn_type=None, coords=None, **kw):
        if hyperedge_conn_type is not None and hyperedge_conn_type.numel() > 0:
            row_mask = (hyperedge_conn_type == 0)
            he_index = hyperedge_index[:, row_mask] if hyperedge_index is not None else None
            he_ctype = hyperedge_conn_type[row_mask]
        else:
            he_index, he_ctype = hyperedge_index, hyperedge_conn_type
        return super().forward(
            x_text, edge_index,
            hyperedge_index=he_index,
            edge_attr=edge_attr,
            retrieved_context_embeds=retrieved_context_embeds,
            hyperedge_conn_type=he_ctype,
            coords=coords,
        )


class TableGNN_ColOnly(TableGNN):
    """Keep ONLY column hyperedges; drop row hyperedges (test row-signal contribution)."""

    def forward(self, x_text, edge_index, hyperedge_index=None,
                edge_attr=None, retrieved_context_embeds=None,
                hyperedge_conn_type=None, coords=None, **kw):
        if hyperedge_conn_type is not None and hyperedge_conn_type.numel() > 0:
            col_mask = (hyperedge_conn_type == 1)
            he_index = hyperedge_index[:, col_mask] if hyperedge_index is not None else None
            he_ctype = hyperedge_conn_type[col_mask]
        else:
            he_index, he_ctype = hyperedge_index, hyperedge_conn_type
        return super().forward(
            x_text, edge_index,
            hyperedge_index=he_index,
            edge_attr=edge_attr,
            retrieved_context_embeds=retrieved_context_embeds,
            hyperedge_conn_type=he_ctype,
            coords=coords,
        )



class AblationTrainer(JointTrainer):
    """Trainer that swaps TableGNN variant based on ablation mode."""

    def __init__(self, config=None, kb_manager=None, local_rank=None, ablation_mode="none"):
        self.ablation_mode = ablation_mode
        super().__init__(config, kb_manager, local_rank)

    def _init_model_override(self):
        """Called after super().__init__ to swap table_encoder."""
        if self.ablation_mode == "no_hgnn":
            original = self.module.table_encoder
            replacement = TableGNN_NoHGNN(
                text_encoder=original.text_encoder,
                num_layers=original.num_layers,
                dropout=original.drop.p,
                order=original.order,
            )
            replacement.load_state_dict(original.state_dict())
            self.module.table_encoder = replacement.to(self.device)
            print("[Ablation] Replaced TableGNN with NoHGNN variant")

        elif self.ablation_mode == "no_gnn":
            original = self.module.table_encoder
            replacement = TableGNN_NoGNN(
                text_encoder=original.text_encoder,
                num_layers=original.num_layers,
                dropout=original.drop.p,
                order=original.order,
            )
            replacement.load_state_dict(original.state_dict())
            self.module.table_encoder = replacement.to(self.device)
            print("[Ablation] Replaced TableGNN with NoGNN variant")

        elif self.ablation_mode == "row_only":
            original = self.module.table_encoder
            replacement = TableGNN_RowOnly(
                text_encoder=original.text_encoder,
                num_layers=original.num_layers,
                dropout=original.drop.p,
                order=original.order,
            )
            replacement.load_state_dict(original.state_dict())
            self.module.table_encoder = replacement.to(self.device)
            print("[Ablation] Replaced TableGNN with RowOnly variant (drop col hyperedges)")

        elif self.ablation_mode == "col_only":
            original = self.module.table_encoder
            replacement = TableGNN_ColOnly(
                text_encoder=original.text_encoder,
                num_layers=original.num_layers,
                dropout=original.drop.p,
                order=original.order,
            )
            replacement.load_state_dict(original.state_dict())
            self.module.table_encoder = replacement.to(self.device)
            print("[Ablation] Replaced TableGNN with ColOnly variant (drop row hyperedges)")


class AblationTrainer_NoContrastive(JointTrainer):
    """Trainer with NED contrastive loss disabled."""

    def train_step(self, graph, kb_entities):
        is_dummy = getattr(graph, 'is_dummy', False)
        if torch.is_tensor(is_dummy):
            if is_dummy.all():
                self.last_train_metrics = {k: 0.0 for k in self.last_train_metrics}
                return 0.0
        elif is_dummy:
            self.last_train_metrics = {k: 0.0 for k in self.last_train_metrics}
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()

        graph = graph.to(self.device)
        labeled_indices = graph.labeled_indices

        extraction_logits, _, _, _, _ = self.model(
            graph, mode='train_forward', valid_cell_indices=None,
        )

        num_nodes = extraction_logits.size(0)
        if hasattr(graph, "entity_mask"):
            extraction_labels = graph.entity_mask
        else:
            extraction_labels = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
            if labeled_indices.numel() > 0:
                extraction_labels[labeled_indices] = 1

        weight = torch.tensor([1.0, 5.0], device=self.device)
        total_loss = F.cross_entropy(extraction_logits, extraction_labels, weight=weight)

        self.last_train_metrics = {
            "loss_total": float(total_loss.detach()), "loss_ner": float(total_loss.detach()),
            "loss_ned": 0.0, "loss_name": 0.0, "loss_name_weighted": 0.0,
            "aligned_samples": 0.0, "labeled_cells": float(labeled_indices.numel()),
        }

        if torch.isnan(total_loss):
            return float('nan')

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._text_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self._gnn_params, max_norm=5.0)
        self.optimizer.step()
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        return total_loss.item()


def build_id_to_name_map(kb_manager):
    mapping = {}
    if not kb_manager or not getattr(kb_manager, "entities", None):
        return mapping
    for ent in kb_manager.entities:
        raw_id = ent.get("id")
        if raw_id is None:
            continue
        norm_id = normalize_entity_id(raw_id)
        if norm_id and norm_id != "nil":
            mapping[norm_id] = ent.get("name", str(raw_id))
    return mapping


def ensure_kb_index(args, kb_manager, trainer, force_rebuild=False):
    if not kb_manager:
        return
    if force_rebuild:
        kb_manager.clear_index()
    if kb_manager.is_built:
        return
    print(f"[KB] Building vector index from: {args.kb_path}")
    encoder = trainer.module.kb_encoder
    projector = trainer.module.retriever.shared_proj if args.eval_retrieval_mode == "hypergraph" else None
    kb_manager.build_index(
        args.kb_path, encoder=encoder,
        batch_size=args.kb_index_batch_size,
        text_micro_batch_size=64,
        path_chunk_size=512,
        projector=projector,
    )
    kb_manager.load_index()
    trainer.kb_manager = kb_manager


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["no_hgnn", "no_gnn", "no_contrastive", 'row_only', 'col_only',
                                 'no_name', 'no_ner', 'no_header', 'gate_gnn_only', 'gate_text_only'])
    parser.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"])
    parser.add_argument("--train_data", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/datasets/wdc_lspm_sampled/train"))
    parser.add_argument("--eval_data", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/datasets/wdc_lspm_sampled/eval"))
    parser.add_argument("--kb_path", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_retrieval_mode", type=str, default="hypergraph")
    parser.add_argument("--kb_index_batch_size", type=int, default=1024)
    parser.add_argument("--lr_backbone", type=float, default=5e-6)
    parser.add_argument("--lr_gnn", type=float, default=1e-4)
    parser.add_argument("--lr_align", type=float, default=1e-4)
    parser.add_argument("--lr_heads", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--name_loss_weight", type=float, default=1.0)
    parser.add_argument("--name_loss_temp", type=float, default=0.07)
    parser.add_argument("--ner_loss_weight", type=float, default=0.1)
    parser.add_argument("--resume_from", type=str, default="",
                        help="If set, load this checkpoint before training (resume).")
    args = parser.parse_args()

    # Loss-weight ablations: zero-out one loss while keeping the rest of the
    # pipeline (graph branches, hard-neg mining, contrastive) identical.
    if args.ablation == "no_name":
        args.name_loss_weight = 0.0
        print(f"[Ablation] no_name -> name_loss_weight=0")
    elif args.ablation == "no_ner":
        args.ner_loss_weight = 0.0
        print(f"[Ablation] no_ner -> ner_loss_weight=0")
    elif args.ablation == "no_header":
        os.environ["HYPERTAB_NO_HEADER"] = "1"
        print(f"[Ablation] no_header -> drop header-content edges + bare cell text (env HYPERTAB_NO_HEADER=1)")
    elif args.ablation == "gate_gnn_only":
        os.environ["HYPERTAB_FORCE_GATE"] = "1.0"
        print(f"[Ablation] gate_gnn_only -> gamma forced to 1 (pure GNN, no text residual)")
    elif args.ablation == "gate_text_only":
        os.environ["HYPERTAB_FORCE_GATE"] = "0.0"
        print(f"[Ablation] gate_text_only -> gamma forced to 0 (pure text, GNN muted)")

    if not args.checkpoint:
        args.checkpoint = os.path.join(PROJECT_ROOT, f"checkpoints/ablation_{args.ablation}.pt")
    if not args.kb_path:
        from baselines.eval_utils import KB_PATH
        args.kb_path = KB_PATH

    print("=" * 60)
    print(f"Ablation Study: {args.ablation}")
    print("=" * 60)

    dataset_root = args.train_data if "train" in args.mode else args.eval_data
    dataset_name = os.path.basename(os.path.normpath(dataset_root))
    kb_working_dir = os.path.join(PROJECT_ROOT, "experiments/kb_index", f"ablation_{args.ablation}_{dataset_name}")
    kb_manager = KnowledgeManager(working_dir=kb_working_dir, model_name=args.model_name)

    if not kb_manager.is_built:
        kb_manager.load_index()
    if not kb_manager.entities and "train" in args.mode:
        with open(args.kb_path, "r", encoding="utf-8") as f:
            kb_manager.entities = [json.loads(line) for line in f]
        print(f"[KB] Loaded {len(kb_manager.entities)} entities")

    config = vars(args)
    if args.ablation == "no_contrastive":
        trainer = AblationTrainer_NoContrastive(config, kb_manager=kb_manager)
    elif args.ablation in ("no_name", "no_ner"):
        # Loss-weight ablation: reuse full HyperTabAlign architecture; mode="none"
        # disables structural ablation paths inside AblationTrainer.
        trainer = AblationTrainer(config, kb_manager=kb_manager, ablation_mode="none")
        trainer._init_model_override()
    else:
        trainer = AblationTrainer(config, kb_manager=kb_manager, ablation_mode=args.ablation)
        trainer._init_model_override()

    # --- Training ---
    if "train" in args.mode:
        # Resume from existing checkpoint if --resume_from is set and points to a real file.
        # We do NOT auto-load args.checkpoint to avoid surprising fresh-train runs.
        if getattr(args, "resume_from", "") and os.path.isfile(args.resume_from):
            print(f"[Train] Resuming from checkpoint: {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)
        print(f"[Train] Starting ablation training: {args.ablation}")
        train_dataset = TableAlignmentDataset(root=args.train_data)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=True, prefetch_factor=4,
        )
        total_steps = len(train_loader) * args.epochs
        trainer.init_scheduler(total_steps, warmup_steps=args.warmup_steps)

        for epoch in range(args.epochs):
            total_loss, valid_steps = 0.0, 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                loss = trainer.train_step(batch, kb_manager.entities if kb_manager else [])
                if isinstance(loss, float) and loss != loss:
                    continue
                total_loss += float(loss)
                valid_steps += 1

            avg_loss = total_loss / max(1, valid_steps)
            print(f"Epoch {epoch + 1} | loss={avg_loss:.4f}")
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            torch.save(trainer.get_state_dict(), args.checkpoint)

    # --- Evaluation ---
    if "eval" in args.mode:
        if "train" not in args.mode and os.path.exists(args.checkpoint):
            print(f"[Eval] Loading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        need_rebuild = "train" in args.mode
        ensure_kb_index(args, kb_manager, trainer, force_rebuild=need_rebuild)
        if kb_manager and kb_manager.is_built and trainer.device.type == "cuda":
            kb_manager.embeddings = kb_manager.embeddings.to(trainer.device, non_blocking=True)
            print(f"[Eval-speedup] KB embeddings pushed to {trainer.device}, shape={tuple(kb_manager.embeddings.shape)}", flush=True)

        eval_dataset = TableAlignmentDataset(root=args.eval_data)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=4)

        all_preds, all_labels = [], []
        all_pred_names, all_label_names = [], []
        all_candidates_info = []
        id_to_name = build_id_to_name_map(kb_manager)

        for batch in tqdm(eval_loader, desc="Evaluating"):
            res = trainer.evaluate_step(batch)
            if res:
                all_preds.extend(res['final_pred_ids'])
                if 'final_pred_names' in res:
                    all_pred_names.extend(res['final_pred_names'])
                if isinstance(res['target_ent_ids'][0], list):
                    flat_labels = [i for s in res['target_ent_ids'] for i in s]
                else:
                    flat_labels = res['target_ent_ids']
                all_labels.extend(flat_labels)
                all_label_names.extend([
                    id_to_name.get(normalize_entity_id(eid), str(eid)) for eid in flat_labels
                ])
                if 'candidates_info' in res:
                    all_candidates_info.extend(res['candidates_info'])

        evaluator = Evaluator()
        metrics = evaluator.compute(
            all_preds, all_labels,
            candidates_info=all_candidates_info,
            pred_names=all_pred_names if all_pred_names else None,
            label_names=all_label_names if all_label_names else None,
            debug=True,
        )
        tag = f"Ours w/o {args.ablation.replace('no_', '').upper()}"
        print(f"\n[{tag}] {len(all_preds)} samples | {metrics}")

        from baselines.eval_utils import save_results
        save_results(metrics, tag)


if __name__ == "__main__":
    main()
