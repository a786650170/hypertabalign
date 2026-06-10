#!/usr/bin/env python3
"""
主入口：HyperGraphRAG 表格实体对齐 — 训练 / 评估 / DDP / 本地 KB 索引 / Early Stopping。
工作目录应为项目根（与 README 一致）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import transformers
from transformers.utils import import_utils

import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from data.dataset import TableAlignmentDataset
from evaluation.metrics import Evaluator
from models.knowledge.manager import KnowledgeManager
from models.knowledge.normalizer import normalize_entity_id
from training.trainer import JointTrainer


def is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None, 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    if is_main_process():
        print(f"[Rank 0] DDP Initialized.")
    return local_rank, dist.get_world_size()


def destroy_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_dataset(root: str, augment: bool = False):
    # Auto-dispatch by hallmark files.
    if os.path.exists(os.path.join(root, "tables.json.gz")):
        from data.wikitables_dataset import WikiTablesAlignmentDataset
        return WikiTablesAlignmentDataset(root=root, augment=augment)
    if os.path.exists(os.path.join(root, "extracted", "semtab_2019_dbpedia_2016-10")):
        from data.semtab_dataset import SemTabAlignmentDataset
        round_num = int(os.environ.get("SEMTAB_ROUND", "2"))
        return SemTabAlignmentDataset(root=root, round_num=round_num, augment=augment)
    return TableAlignmentDataset(root=root, augment=augment)


def get_kb_projector(trainer: JointTrainer, eval_retrieval_mode: str):
    if eval_retrieval_mode == "hypergraph":
        return trainer.module.retriever.shared_proj
    return None


def reset_alignment_head_on_resume(trainer: JointTrainer):
    """Resume 时重置对齐头与温度，避免旧 checkpoint 与当前结构不兼容。"""
    sp = trainer.module.retriever.shared_proj
    torch.nn.init.xavier_uniform_(sp.weight)
    if sp.bias is not None:
        torch.nn.init.zeros_(sp.bias)
    with torch.no_grad():
        trainer.module.retriever.temperature.fill_(0.07)
    if is_main_process():
        print("[Train] reset_alignment_head_on_resume: shared_proj re-init, temperature=0.07")


def load_kb_entities(kb_path: str):
    entities = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entities.append(json.loads(line))
    return entities


def build_id_to_name_map(kb_manager: KnowledgeManager):
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


def _flatten_x_text(graph) -> list:
    xt = getattr(graph, "x_text", None)
    if not xt:
        return []
    if isinstance(xt[0], str):
        return list(xt)
    return [t for sub in xt for t in sub]


def cell_texts_for_labeled(graph, labeled_indices: torch.Tensor) -> list:
    flat = _flatten_x_text(graph)
    if labeled_indices.numel() == 0 or not flat:
        return []
    out = []
    for i in labeled_indices.detach().cpu().tolist():
        out.append(flat[i] if i < len(flat) else "")
    return out


def ensure_kb_entities(kb_manager: KnowledgeManager, kb_path: str):
    if kb_manager.entities:
        return
    if os.path.isfile(kb_path):
        kb_manager.entities = load_kb_entities(kb_path)
        if is_main_process():
            print(f"[KB] Loaded {len(kb_manager.entities)} entities from {kb_path}")


def ensure_kb_index(args, kb_manager: KnowledgeManager, trainer: JointTrainer, force_rebuild: bool):
    if not args.use_local_kb or not kb_manager:
        return
    if force_rebuild:
        kb_manager.clear_index()
    if kb_manager.is_built:
        if is_main_process():
            print(f"[KM] 索引已存在，跳过构建: {kb_manager.working_dir}")
        return
    if is_main_process():
        print(f"[KB] Building vector index from: {args.kb_path}")
    ensure_kb_entities(kb_manager, args.kb_path)
    encoder = trainer.module.kb_encoder
    projector = get_kb_projector(trainer, args.eval_retrieval_mode)
    kb_manager.build_index(
        args.kb_path,
        encoder=encoder,
        batch_size=args.kb_index_batch_size,
        text_micro_batch_size=args.kb_text_micro_batch_size,
        path_chunk_size=args.kb_path_chunk_size,
        projector=projector,
    )
    kb_manager.load_index()
    trainer.kb_manager = kb_manager
    if is_main_process():
        print(f"[KB] Ready: entities={len(kb_manager.entities)}, emb={kb_manager.embeddings.shape}")


def gather_concat_lists(local_list: list, world_size: int) -> list:
    if not dist.is_initialized() or world_size <= 1:
        return list(local_list)
    gathered = [None] * world_size
    dist.all_gather_object(gathered, list(local_list))
    merged = []
    for part in gathered:
        if part:
            merged.extend(part)
    return merged


def _train_ned_then_name_improved(avg_ned: float, avg_name: float, avg_loss: float, best: dict) -> bool:
    bn, bm, bl = best["ned"], best["name"], best["loss"]
    return (
        avg_ned < bn - 1e-8
        or (abs(avg_ned - bn) <= 1e-8 and avg_name < bm - 1e-8)
        or (abs(avg_ned - bn) <= 1e-8 and abs(avg_name - bm) <= 1e-8 and avg_loss < bl - 1e-8)
    )


def _selection_improved(sel: str, avg_loss: float, avg_ned: float, avg_name: float, best: dict) -> bool:
    if sel == "train_loss":
        return avg_loss < best["train_loss"] - 1e-8
    if sel == "train_ned_then_name":
        return _train_ned_then_name_improved(avg_ned, avg_name, avg_loss, best)
    return False


def _selection_update_best(sel: str, improved: bool, best: dict, avg_loss: float, avg_ned: float, avg_name: float):
    if not improved:
        return
    if sel == "train_loss":
        best["train_loss"] = avg_loss
    elif sel == "train_ned_then_name":
        best["loss"], best["ned"], best["name"] = avg_loss, avg_ned, avg_name


def run_alignment_eval_pass(
    trainer: JointTrainer,
    args,
    kb_manager: KnowledgeManager | None,
    data_root: str,
    world_size: int,
    desc: str,
    max_batches: int = 0,
) -> dict | None:
    """在 data_root 上跑一遍与主 eval 相同的检索评估；仅在 rank0 返回 metrics dict。"""
    if not kb_manager or not kb_manager.is_built:
        if is_main_process():
            print(f"[{desc}] 跳过：KB 索引未就绪（需 --use_local_kb 且已成功 build/load index）")
        return None

    eval_ds = get_dataset(data_root, augment=False)
    eval_sampler = DistributedSampler(eval_ds, shuffle=False) if dist.is_initialized() else None
    nw = min(args.num_workers, 4)
    ev_kw = dict(
        batch_size=args.batch_size,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    if nw > 0:
        ev_kw["persistent_workers"] = True
        ev_kw["prefetch_factor"] = 2
    eval_loader = DataLoader(eval_ds, **ev_kw)

    id_to_name = build_id_to_name_map(kb_manager)
    all_preds, all_labels = [], []
    all_pred_names, all_label_names = [], []
    all_cell_texts = []
    all_cand = []

    it = tqdm(eval_loader, desc=desc, disable=not is_main_process())
    for bi, batch in enumerate(it):
        res = trainer.evaluate_step(batch, rag_rounds=args.rag_eval_rounds)
        if not res:
            continue
        li = batch.labeled_indices
        ctexts = cell_texts_for_labeled(batch, li)

        all_preds.extend(res["final_pred_ids"])
        all_pred_names.extend(res.get("final_pred_names") or [])
        tgt = res["target_ent_ids"]
        if isinstance(tgt[0], (list, tuple)):
            flat_labels = [x for sub in tgt for x in sub]
        else:
            flat_labels = list(tgt)
        all_labels.extend(flat_labels)
        all_label_names.extend([id_to_name.get(normalize_entity_id(eid), str(eid)) for eid in flat_labels])
        if ctexts:
            all_cell_texts.extend(ctexts)
        if "candidates_info" in res:
            all_cand.extend(res["candidates_info"])

        if max_batches and (bi + 1) >= max_batches:
            break

    if dist.is_initialized() and world_size > 1:
        all_preds = gather_concat_lists(all_preds, world_size)
        all_labels = gather_concat_lists(all_labels, world_size)
        all_pred_names = gather_concat_lists(all_pred_names, world_size)
        all_label_names = gather_concat_lists(all_label_names, world_size)
        all_cell_texts = gather_concat_lists(all_cell_texts, world_size)
        all_cand = gather_concat_lists(all_cand, world_size)

    if not is_main_process():
        return None

    ev = Evaluator()
    return ev.compute(
        all_preds,
        all_labels,
        candidates_info=all_cand if all_cand else None,
        pred_names=all_pred_names if all_pred_names else None,
        label_names=all_label_names if all_label_names else None,
        cell_texts=all_cell_texts if all_cell_texts else None,
        debug=args.eval_debug,
    )


def parse_args():
    p = argparse.ArgumentParser(description="HyperGraphRAG — Table Entity Alignment")
    p.add_argument("--mode", type=str, default="train_eval", choices=["train", "eval", "train_eval"])

    p.add_argument("--train_data", type=str, default=os.path.join(PROJECT_ROOT, "data/datasets/wdc_lspm_sampled/train"))
    p.add_argument("--eval_data", type=str, default=os.path.join(PROJECT_ROOT, "data/datasets/wdc_lspm_sampled/eval"))
    p.add_argument("--kb_path", type=str, default="")

    p.add_argument("--checkpoint", type=str, default=os.path.join(PROJECT_ROOT, "checkpoints/wdc_hypergraph.pt"))
    p.add_argument("--fresh_start", action="store_true", help="训练时忽略已有 checkpoint，从头训练")

    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--batch_size", type=int, default=128, help="与近期 WDC 三卡实验一致（每卡 batch）")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="按 --selection_metric 主指标无提升时的 early stopping 容忍 epoch 数",
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_train_steps", type=int, default=0, help=">0 时每 epoch 最多训练步数（调试）")

    p.add_argument("--use_local_kb", action="store_true")
    p.add_argument("--rebuild_kb_index", action="store_true")
    p.add_argument("--kb_working_dir", type=str, default="", help="KB 索引目录；空则 experiments/kb_index/train_hypergraph")
    p.add_argument("--kb_index_batch_size", type=int, default=1024)
    p.add_argument("--kb_text_micro_batch_size", type=int, default=64)
    p.add_argument("--kb_path_chunk_size", type=int, default=512)

    p.add_argument("--eval_retrieval_mode", type=str, default="hypergraph", choices=["hypergraph", "direct"])
    p.add_argument("--rag_eval_rounds", type=int, default=0)

    p.add_argument("--gnn_hidden_dim", type=int, default=768)
    p.add_argument("--retrieval_dim", type=int, default=256)
    p.add_argument("--gnn_layers", type=int, default=2)
    p.add_argument("--gnn_dropout", type=float, default=0.1)
    p.add_argument("--gnn_order", type=str, default="local_first")

    p.add_argument("--ner_loss_weight", type=float, default=0.1)
    p.add_argument("--name_loss_weight", type=float, default=0.3)
    p.add_argument("--name_loss_temp", type=float, default=0.07)
    p.add_argument("--ner_label_smoothing", type=float, default=0.0, help="NER 分类平滑；默认 0 关闭，二分类+class weight 场景一般无收益（保留 CLI 便于消融）")

    p.add_argument("--lr_backbone", type=float, default=5e-6)
    p.add_argument("--lr_gnn", type=float, default=1.5e-4, help="超图/GAT 分支；略升以加快结构分支离开平台期")
    p.add_argument("--lr_align", type=float, default=2e-4, help="alignment_proj + shared_proj；对比空间主瓶颈，略升加速 NED/Name")
    p.add_argument("--lr_heads", type=float, default=7e-4, help="KBEncoder 等 head 组；略升加快 path/name 编码适应")
    p.add_argument("--warmup_steps", type=int, default=120, help="略缩短以更快进入目标 LR（仍 cosine 收尾）")
    p.add_argument("--num_train_negatives", type=int, default=96, help="每步训练负例数（略增强化对比信号）")
    p.add_argument("--hard_negative_topk", type=int, default=48, help="KB 近邻候选深度（略增利于难负例多样性）")
    p.add_argument("--hard_negative_random_ratio", type=float, default=0.25, help="难负例中随机补齐比例，降低伪难例误伤")
    p.add_argument(
        "--selection_metric",
        type=str,
        default="train_ned_then_name",
        choices=["train_loss", "train_ned_then_name"],
        help="best checkpoint 与 early stopping 的主指标（基于训练集统计量）",
    )

    p.add_argument("--reset_alignment_head_on_resume", action="store_true")
    p.add_argument("--eval_debug", action="store_true", help="评估时打印部分样本 debug")
    p.add_argument(
        "--eval_best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train_eval 结束后是否加载 *_best.pt 评 test（默认启用；与 --selection_metric 一致）。"
        "若 *_best.pt 不存在则自动退回主 checkpoint。",
    )

    return p.parse_args()


def main():
    args = parse_args()
    local_rank, world_size = init_distributed()

    if not args.kb_path:
        from baselines.eval_utils import KB_PATH
        args.kb_path = KB_PATH

    if not args.kb_working_dir:
        args.kb_working_dir = os.path.join(PROJECT_ROOT, "experiments", "kb_index", "train_hypergraph")

    if is_main_process():
        print(f"[KB] Working directory: {args.kb_working_dir}")

    kb_manager = KnowledgeManager(working_dir=args.kb_working_dir, model_name=args.model_name) if args.use_local_kb else None

    if kb_manager and not kb_manager.entities and "train" in args.mode:
        ensure_kb_entities(kb_manager, args.kb_path)

    config = vars(args)
    trainer = JointTrainer(config, kb_manager=kb_manager, local_rank=local_rank)
    device = trainer.device
    best_path = os.path.splitext(args.checkpoint)[0] + "_best.pt"

    if args.reset_alignment_head_on_resume:
        reset_alignment_head_on_resume(trainer)

    if "train" in args.mode:
        if args.fresh_start:
            if is_main_process():
                print("[Train] 🔥 Fresh start: training from scratch, ignoring old checkpoint.", flush=True)
        elif os.path.isfile(args.checkpoint):
            if is_main_process():
                print(f"[Train] Resuming from {args.checkpoint}...", flush=True)
            trainer.load_checkpoint(args.checkpoint)

    if "train" in args.mode:
        if is_main_process():
            print(f"[Train] Starting training on: {args.train_data}")
        train_ds = get_dataset(args.train_data, augment=True)
        train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
        nw = args.num_workers
        loader_kw = dict(
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )
        if nw > 0:
            loader_kw["persistent_workers"] = True
            loader_kw["prefetch_factor"] = 2
        train_loader = DataLoader(train_ds, **loader_kw)

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * args.epochs
        if args.max_train_steps > 0:
            total_steps = min(total_steps, args.max_train_steps * max(1, args.epochs))
        trainer.init_scheduler(total_steps, warmup_steps=args.warmup_steps)

        sel_metric = args.selection_metric
        if is_main_process():
            print(
                f"[Train] early stopping / *_best.pt 依据 selection_metric={sel_metric}",
                flush=True,
            )

        if args.use_local_kb and kb_manager:
            ensure_kb_entities(kb_manager, args.kb_path)
            if is_main_process():
                print("[Train] 加载或构建 KB 索引（难负例与 eval 检索需要）...", flush=True)
            ensure_kb_index(args, kb_manager, trainer, force_rebuild=False)

        best = {
            "train_loss": float("inf"),
            "loss": float("inf"),
            "ned": float("inf"),
            "name": float("inf"),
        }
        patience_ctr = 0

        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            trainer.model.train()
            total_loss = 0.0
            total_loss_ner = 0.0
            total_loss_ned = 0.0
            total_loss_name = 0.0
            total_loss_name_w = 0.0
            total_aligned = 0.0
            total_labeled = 0.0
            valid_steps = 0

            it = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not is_main_process())
            for batch in it:
                kb_ent = kb_manager.entities if kb_manager else []
                loss = trainer.train_step(batch, kb_ent)
                if isinstance(loss, float) and loss != loss:
                    continue
                m = getattr(trainer, "last_train_metrics", {})
                total_loss += float(m.get("loss_total", 0.0))
                total_loss_ner += float(m.get("loss_ner", 0.0))
                total_loss_ned += float(m.get("loss_ned", 0.0))
                total_loss_name += float(m.get("loss_name", 0.0))
                total_loss_name_w += float(m.get("loss_name_weighted", 0.0))
                total_aligned += float(m.get("aligned_samples", 0.0))
                total_labeled += float(m.get("labeled_cells", 0.0))
                valid_steps += 1

                if is_main_process() and valid_steps % 10 == 0:
                    s = valid_steps
                    avg_ned = total_loss_ned / s
                    avg_name = total_loss_name / s
                    avg_ner = total_loss_ner / s
                    avg_tot = total_loss / s
                    cur_lr = trainer.optimizer.param_groups[0]['lr']
                    print(
                        f"  [step {s}] NED={avg_ned:.4f} Name={avg_name:.4f} "
                        f"NER={avg_ner:.4f} total={avg_tot:.4f} lr_text={cur_lr:.2e}",
                        flush=True,
                    )

                if args.max_train_steps > 0 and valid_steps >= args.max_train_steps:
                    break

            vs = max(1, valid_steps)
            avg_loss = total_loss / vs
            avg_ner = total_loss_ner / vs
            avg_ned = total_loss_ned / vs
            avg_name = total_loss_name / vs
            avg_name_w = total_loss_name_w / vs
            avg_aligned = total_aligned / vs
            avg_labeled = total_labeled / vs

            if dist.is_initialized():
                t = torch.tensor(
                    [avg_loss, avg_ner, avg_ned, avg_name, avg_name_w, avg_aligned, avg_labeled, float(valid_steps)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t /= world_size
                avg_loss, avg_ner, avg_ned, avg_name, avg_name_w, avg_aligned, avg_labeled = (
                    float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4]), float(t[5]), float(t[6])
                )

            if is_main_process():
                print(f"Epoch {epoch + 1} | loss={avg_loss:.4f}")
                print(f"  NER={avg_ner:.4f}  NED={avg_ned:.4f}  Name={avg_name:.4f} (w={avg_name_w:.4f})")
                print(f"  labeled_cells={avg_labeled:.1f}  aligned_samples={avg_aligned:.1f}")
                os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
                torch.save(trainer.get_state_dict(), args.checkpoint)
                print(f"  Saved: {args.checkpoint}")

            improved = _selection_improved(sel_metric, avg_loss, avg_ned, avg_name, best)

            if improved:
                _selection_update_best(sel_metric, True, best, avg_loss, avg_ned, avg_name)
                patience_ctr = 0
                if is_main_process():
                    torch.save(trainer.get_state_dict(), best_path)
                    if sel_metric == "train_loss":
                        print(
                            f"  ★ Best checkpoint: {best_path} "
                            f"(metric=train_loss, train_loss={best['train_loss']:.4f})",
                            flush=True,
                        )
                    else:
                        print(
                            f"  ★ Best checkpoint: {best_path} "
                            f"(metric=train_ned_then_name, NED={best['ned']:.4f}, "
                            f"Name={best['name']:.4f}, loss={best['loss']:.4f})",
                            flush=True,
                        )
            else:
                patience_ctr += 1

            if dist.is_initialized():
                pc = torch.tensor([patience_ctr], device=device, dtype=torch.long)
                dist.broadcast(pc, src=0)
                patience_ctr = int(pc.item())

            if patience_ctr >= args.patience:
                if is_main_process():
                    print(f"[Train] Early stopping: patience={args.patience} reached at epoch {epoch + 1}.")
                break

    if "eval" in args.mode:
        eval_ckpt = args.checkpoint
        if args.eval_best and os.path.isfile(best_path):
            eval_ckpt = best_path
            if is_main_process():
                print(f"[Eval] 将使用 best 权重: {eval_ckpt}")

        if "train" not in args.mode:
            if os.path.isfile(eval_ckpt):
                if is_main_process():
                    print(f"[Eval] Loading checkpoint: {eval_ckpt}")
                trainer.load_checkpoint(eval_ckpt)
        elif args.eval_best and os.path.isfile(best_path):
            if is_main_process():
                print(f"[Eval] Loading best checkpoint: {eval_ckpt}")
            trainer.load_checkpoint(eval_ckpt)
        elif is_main_process():
            print(
                "[Eval] 使用训练结束时的内存权重（与上次 train_eval 一致）；"
                "默认会在存在 *_best.pt 时加载 best；若要用最后一轮权重请加 --no-eval-best",
                flush=True,
            )

        if kb_manager:
            ensure_kb_entities(kb_manager, args.kb_path)
            need_rebuild = args.rebuild_kb_index or (not kb_manager.is_built)
            if "train" in args.mode:
                need_rebuild = True
            ensure_kb_index(args, kb_manager, trainer, force_rebuild=need_rebuild)

        if is_main_process():
            print(f"[Eval] Starting evaluation on: {args.eval_data}")

        trainer.model.eval()
        metrics = run_alignment_eval_pass(
            trainer,
            args,
            kb_manager,
            args.eval_data,
            world_size,
            desc="Evaluating",
            max_batches=0,
        )
        if is_main_process():
            if metrics is not None:
                print(f"\n[Eval] metrics | {metrics}")
                try:
                    from baselines.eval_utils import save_results
                    save_results(metrics, "HyperGraphRAG (main)")
                except Exception as e:
                    print(f"[Eval] save_results skipped: {e}")
            else:
                print("[Eval] 无有效指标（检查 --use_local_kb 与 KB 索引）")

    destroy_distributed()


if __name__ == "__main__":
    main()
