import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.knowledge.normalizer import normalize_entity_id
from torch.nn.parallel import DistributedDataParallel as DDP
from models.unified_model import HyperGraphRAGModel
from models.knowledge.manager import KnowledgeManager


class JointTrainer(nn.Module):
    def __init__(self, config=None, kb_manager: KnowledgeManager = None, local_rank=None):
        super().__init__()
        self.local_rank = local_rank
        if local_rank is not None:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.kb_manager = kb_manager

        if config is None:
            config = {}

        self.model = HyperGraphRAGModel(config)
        self.model.to(self.device)

        if local_rank is not None:
            if self.local_rank == 0:
                print(f"[Rank {local_rank}] Initializing DDP...")
            self.model = DDP(
                self.model, device_ids=[local_rank],
                output_device=local_rank, find_unused_parameters=True,
            )
            self._is_ddp = True
            self.module = self.model.module
        else:
            self._is_ddp = False
            self.module = self.model

        self.eval_retrieval_mode = config.get('eval_retrieval_mode', 'hypergraph')
        self.ner_loss_weight = float(config.get('ner_loss_weight', 0.1))
        self.name_loss_weight = float(config.get('name_loss_weight', 0.3))
        self.name_loss_temp = float(config.get('name_loss_temp', 0.07))
        self.num_train_negatives = int(config.get('num_train_negatives', 64))
        self.hard_negative_topk = int(config.get('hard_negative_topk', 32))
        self.grad_clip_text = float(config.get('grad_clip_text', 1.0))
        self.grad_clip_gnn = float(config.get('grad_clip_gnn', 5.0))
        self.grad_clip_align = float(config.get('grad_clip_align', 8.0))
        self.grad_clip_heads = float(config.get('grad_clip_heads', 12.0))
        self.ner_label_smoothing = float(config.get('ner_label_smoothing', 0.0))
        self.hard_negative_random_ratio = float(config.get('hard_negative_random_ratio', 0.25))

        text_params_list = list(self.module.shared_text_encoder.parameters())
        text_params_set = set(text_params_list)

        gnn_modules = [
            self.module.table_encoder.pre_proj,
            self.module.table_encoder.edge_type_embed,
            self.module.table_encoder.gat_layers,
            self.module.table_encoder.row_hgnn_layers,
            self.module.table_encoder.col_hgnn_layers,
            self.module.table_encoder.rc_gates,
            self.module.table_encoder.norms,
            self.module.table_encoder.extraction_head,
        ]
        gnn_params_list = []
        gnn_params_set = set()
        for m in gnn_modules:
            for p in m.parameters():
                if p not in text_params_set and p not in gnn_params_set:
                    gnn_params_list.append(p)
                    gnn_params_set.add(p)

        align_modules = [
            self.module.table_encoder.alignment_proj,
            self.module.retriever.shared_proj,
        ]
        temperature_param = self.module.retriever.temperature
        align_params_list = []
        align_params_set = {temperature_param}
        for m in align_modules:
            for p in m.parameters():
                if p not in text_params_set and p not in gnn_params_set and p not in align_params_set:
                    align_params_list.append(p)
                    align_params_set.add(p)

        head_params = [
            p for p in self.module.parameters()
            if p not in text_params_set and p not in gnn_params_set and p not in align_params_set
        ]

        self._text_params = text_params_list
        self._gnn_params = gnn_params_list
        self._align_params = align_params_list + [temperature_param]
        self._head_params = head_params

        lr_backbone = float(config.get('lr_backbone', 5e-6))
        lr_gnn = float(config.get('lr_gnn', 1e-4))
        lr_align = float(config.get('lr_align', 1e-4))
        lr_heads = float(config.get('lr_heads', 5e-4))

        self.optimizer = torch.optim.AdamW([
            {'params': text_params_list, 'lr': lr_backbone, 'weight_decay': 0.01},
            {'params': gnn_params_list, 'lr': lr_gnn, 'weight_decay': 0.01},
            {'params': align_params_list, 'lr': lr_align, 'weight_decay': 0.01},
            {'params': [temperature_param], 'lr': lr_align, 'weight_decay': 0.0},
            {'params': head_params, 'lr': lr_heads, 'weight_decay': 1e-4},
        ], eps=1e-6)

        if self.local_rank is None or self.local_rank == 0:
            n_text = sum(p.numel() for p in text_params_list)
            n_gnn = sum(p.numel() for p in gnn_params_list)
            n_align = sum(p.numel() for p in align_params_list)
            n_head = sum(p.numel() for p in head_params)
            print(f"[Optimizer] 5 groups: text={n_text/1e6:.1f}M(lr={lr_backbone}) "
                  f"gnn={n_gnn/1e6:.2f}M(lr={lr_gnn}) align={n_align/1e6:.3f}M(lr={lr_align}) "
                  f"temperature(wd=0) heads={n_head/1e6:.2f}M(lr={lr_heads})")

        self.last_train_metrics = {
            "loss_total": 0.0, "loss_ner": 0.0, "loss_ned": 0.0,
            "loss_name": 0.0, "loss_name_weighted": 0.0,
            "aligned_samples": 0.0, "labeled_cells": 0.0,
        }

    def init_scheduler(self, total_steps: int, warmup_steps: int = 200):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        if self.local_rank is None or self.local_rank == 0:
            print(f"[Scheduler] Warmup {warmup_steps} steps, cosine decay over {total_steps} total steps.")

    def _build_id_to_name_map(self):
        if hasattr(self, '_id_to_name_cache') and self._id_to_name_cache:
            return self._id_to_name_cache
        id_to_name = {}
        if not self.kb_manager or not self.kb_manager.entities:
            return id_to_name
        for ent in self.kb_manager.entities:
            raw_id = ent.get("id")
            if raw_id is None:
                continue
            key = normalize_entity_id(raw_id)
            if key and key != "nil":
                id_to_name[key] = ent.get("name", str(raw_id))
        self._id_to_name_cache = id_to_name
        print(f"[Trainer] id->name 缓存构建完成，共 {len(id_to_name)} 条。")
        return id_to_name

    def _multi_positive_contrastive_loss(self, logits, positive_mask, bidirectional=False):
        positive_mask = positive_mask.bool()
        row_has_positive = positive_mask.any(dim=1)
        if not torch.any(row_has_positive):
            return logits.new_tensor(0.0)

        masked_positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        log_prob = torch.logsumexp(masked_positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
        loss = -log_prob[row_has_positive].mean()

        if bidirectional:
            reverse_mask = positive_mask.t()
            col_has_positive = reverse_mask.any(dim=1)
            masked_reverse_logits = logits.t().masked_fill(~reverse_mask, float("-inf"))
            reverse_log_prob = (
                torch.logsumexp(masked_reverse_logits, dim=1) - torch.logsumexp(logits.t(), dim=1)
            )
            reverse_loss = (
                -reverse_log_prob[col_has_positive].mean()
                if torch.any(col_has_positive) else logits.new_tensor(0.0)
            )
            loss = 0.5 * (loss + reverse_loss)

        return loss

    def _match_entities(self, labeled_indices, target_ent_ids, kb_entities):
        if isinstance(target_ent_ids[0], (list, tuple)):
            target_ent_ids = [eid for sublist in target_ent_ids for eid in sublist]

        if not hasattr(self, "ent_id_to_idx_cache") or len(self.ent_id_to_idx_cache) != len(kb_entities):
            self.ent_id_to_idx_cache = {}
            for i, e in enumerate(kb_entities):
                raw_id = str(e['id'])
                self.ent_id_to_idx_cache[raw_id] = i
                clean_id = raw_id.split("/")[-1].strip()
                self.ent_id_to_idx_cache[clean_id] = i
                self.ent_id_to_idx_cache[raw_id.lower()] = i
                self.ent_id_to_idx_cache[clean_id.lower()] = i

        ent_id_to_idx = self.ent_id_to_idx_cache
        valid_targets = []
        final_valid_indices = []

        for i, eid in enumerate(target_ent_ids):
            eid_str = str(eid).strip()
            eid_clean = eid_str.split("/")[-1]
            match_idx = None
            if eid_str in ent_id_to_idx:
                match_idx = ent_id_to_idx[eid_str]
            elif eid_clean in ent_id_to_idx:
                match_idx = ent_id_to_idx[eid_clean]
            elif eid_str.lower() in ent_id_to_idx:
                match_idx = ent_id_to_idx[eid_str.lower()]
            elif eid_clean.lower() in ent_id_to_idx:
                match_idx = ent_id_to_idx[eid_clean.lower()]
            if match_idx is not None:
                valid_targets.append(match_idx)
                final_valid_indices.append(labeled_indices[i].item())

        return final_valid_indices, valid_targets

    def _sample_hard_negatives(self, positive_kb_indices, kb_entities, num_negatives=32):
        import random
        pos_set = set(positive_kb_indices)
        total = len(kb_entities)
        if total == 0:
            return [], []
        neg_indices = []
        seen = set()

        hard_quota = max(0, min(num_negatives, int(round(num_negatives * (1.0 - self.hard_negative_random_ratio)))))
        rand_quota = max(0, num_negatives - hard_quota)

        # 优先使用索引检索出的近邻实体作为难负例。
        if (
            self.kb_manager is not None
            and getattr(self.kb_manager, "is_built", False)
            and getattr(self.kb_manager, "embeddings", None) is not None
            and total == self.kb_manager.embeddings.size(0)
            and len(positive_kb_indices) > 0
        ):
            unique_pos = []
            uniq_seen = set()
            for idx in positive_kb_indices:
                if idx not in uniq_seen:
                    uniq_seen.add(idx)
                    unique_pos.append(idx)
            query_embeds = self.kb_manager.embeddings[unique_pos]
            top_k = max(self.hard_negative_topk, 2)
            retrieved = self.kb_manager.query(query_embeds, top_k=top_k)
            cand_rows = retrieved.get("indices")
            if cand_rows is not None:
                for row in cand_rows.tolist():
                    for idx in row:
                        if idx in pos_set or idx in seen:
                            continue
                        neg_indices.append(idx)
                        seen.add(idx)
                        if len(neg_indices) >= hard_quota:
                            break
                    if len(neg_indices) >= hard_quota:
                        break

        # 随机负例用于降低“伪难例”误伤，同样可补齐 hard quota 不足。
        attempts = 0
        need_total = max(num_negatives, len(neg_indices) + rand_quota)
        max_attempts = max(need_total * 8, 128)
        while len(neg_indices) < need_total and attempts < max_attempts:
            idx = random.randint(0, total - 1)
            if idx not in pos_set and idx not in seen:
                neg_indices.append(idx)
                seen.add(idx)
            attempts += 1
        neg_indices = neg_indices[:num_negatives]
        neg_names = [kb_entities[i]['name'] for i in neg_indices]
        neg_paths = [kb_entities[i].get('path', '') for i in neg_indices]
        return neg_names, neg_paths

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
        target_ent_ids = graph.target_ent_ids

        has_targets = labeled_indices.numel() > 0 and kb_entities
        final_valid_indices = []
        valid_targets = []
        if has_targets:
            final_valid_indices, valid_targets = self._match_entities(
                labeled_indices, target_ent_ids, kb_entities
            )

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if final_valid_indices:
                target_indices_tensor = torch.tensor(valid_targets, device=self.device)
                target_names = [kb_entities[idx]['name'] for idx in valid_targets]
                target_paths = [kb_entities[idx].get('path', '') for idx in valid_targets]
                valid_cell_indices = torch.tensor(final_valid_indices, dtype=torch.long, device=self.device)
                neg_names, neg_paths = self._sample_hard_negatives(
                    valid_targets, kb_entities, num_negatives=self.num_train_negatives
                )

                results = self.model(
                    graph,
                    mode='train_forward',
                    valid_cell_indices=valid_cell_indices,
                    target_names=target_names,
                    target_paths=target_paths,
                    neg_names=neg_names,
                    neg_paths=neg_paths,
                )
                extraction_logits, retrieval_logits, neg_logits, labeled_cell_embeds, name_embeds_proj = results
            else:
                extraction_logits, _, _, _, _ = self.model(
                    graph,
                    mode='train_forward',
                    valid_cell_indices=None,
                )
                retrieval_logits = None

            num_nodes = extraction_logits.size(0)
            if hasattr(graph, "entity_mask"):
                extraction_labels = graph.entity_mask
            else:
                extraction_labels = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
                if labeled_indices.numel() > 0:
                    extraction_labels[labeled_indices] = 1

            weight = torch.tensor([1.0, 5.0], device=self.device)
            loss_ner_raw = F.cross_entropy(
                extraction_logits,
                extraction_labels,
                weight=weight,
                label_smoothing=self.ner_label_smoothing,
            )
            loss_ner = self.ner_loss_weight * loss_ner_raw

            metrics = {
                "loss_total": 0.0, "loss_ner": float(loss_ner_raw.detach()),
                "loss_ned": 0.0, "loss_name": 0.0, "loss_name_weighted": 0.0,
                "aligned_samples": 0.0, "labeled_cells": float(labeled_indices.numel()),
            }

            if retrieval_logits is not None and final_valid_indices:
                target_index_values = target_indices_tensor.detach()
                positive_mask = target_index_values.unsqueeze(0) == target_index_values.unsqueeze(1)

                if neg_logits is not None:
                    full_logits = torch.cat([retrieval_logits, neg_logits], dim=1)
                    neg_mask = torch.zeros(
                        positive_mask.size(0), neg_logits.size(1),
                        dtype=torch.bool, device=self.device,
                    )
                    full_mask = torch.cat([positive_mask, neg_mask], dim=1)
                else:
                    full_logits = retrieval_logits
                    full_mask = positive_mask

                loss_ned = self._multi_positive_contrastive_loss(full_logits, full_mask)

                cell_norm = F.normalize(labeled_cell_embeds, p=2, dim=-1)
                name_norm = F.normalize(name_embeds_proj, p=2, dim=-1)
                loss_name_pos = (1.0 - (cell_norm * name_norm).sum(dim=-1)).mean()
                if cell_norm.size(0) > 1:
                    name_logits = torch.matmul(cell_norm, name_norm.t()) / self.name_loss_temp
                    loss_name_nce = self._multi_positive_contrastive_loss(
                        name_logits, positive_mask, bidirectional=True,
                    )
                    loss_name = 0.5 * loss_name_pos + 0.5 * loss_name_nce
                else:
                    loss_name = loss_name_pos

                weighted_name_loss = self.name_loss_weight * loss_name
                total_loss = loss_ner + loss_ned + weighted_name_loss
                metrics.update({
                    "loss_total": float(total_loss.detach()),
                    "loss_ned": float(loss_ned.detach()),
                    "loss_name": float(loss_name.detach()),
                    "loss_name_weighted": float(weighted_name_loss.detach()),
                    "aligned_samples": float(len(final_valid_indices)),
                })
            else:
                total_loss = loss_ner
                metrics["loss_total"] = float(total_loss.detach())

        self.last_train_metrics = metrics

        if torch.isnan(total_loss):
            if self.local_rank is None or self.local_rank == 0:
                print("Warning: Loss is NaN, skipping step.")
            return float('nan')

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._text_params, max_norm=self.grad_clip_text)
        torch.nn.utils.clip_grad_norm_(self._gnn_params, max_norm=self.grad_clip_gnn)
        torch.nn.utils.clip_grad_norm_(self._align_params, max_norm=self.grad_clip_align)
        torch.nn.utils.clip_grad_norm_(self._head_params, max_norm=self.grad_clip_heads)
        self.optimizer.step()
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        return total_loss.item()

    def evaluate_step(self, graph, rag_rounds: int = 0):
        self.model.eval()
        graph = graph.to(self.device)

        with torch.no_grad():
            extraction_logits, alignment_embeds = self.model(
                graph, mode='encode_table',
            )

            if rag_rounds > 0 and self.kb_manager and self.kb_manager.is_built:
                for _ in range(rag_rounds):
                    query_embeds = self.module.retriever.shared_proj(alignment_embeds)
                    query_embeds = torch.nn.functional.normalize(query_embeds, p=2, dim=-1)
                    retrieved_context = self.kb_manager.query(query_embeds, top_k=3)
                    ctx_embeds = torch.zeros_like(alignment_embeds)
                    if retrieved_context['scores'].numel() > 0:
                        weights = torch.softmax(retrieved_context['scores'], dim=-1)
                        for ni in range(query_embeds.size(0)):
                            cand_idx = retrieved_context['indices'][ni]
                            cand_embs = self.kb_manager.embeddings[cand_idx].to(ctx_embeds.device)
                            ctx_embeds[ni] = (weights[ni].unsqueeze(-1) * cand_embs).sum(dim=0)
                    extraction_logits, alignment_embeds = self.model(
                        graph, mode='encode_table',
                        retrieved_context_embeds=ctx_embeds,
                    )

            final_pred_ids = []
            final_pred_names = []
            all_candidates_info = []

            labeled_indices = graph.labeled_indices
            target_ent_ids = graph.target_ent_ids

            if labeled_indices.numel() == 0:
                return None

            if self.kb_manager and self.kb_manager.is_built:
                if self.eval_retrieval_mode == "hypergraph":
                    labeled_embeds = self.module.retriever.shared_proj(
                        alignment_embeds[labeled_indices]
                    )
                else:
                    flat_x_text = graph.x_text
                    if flat_x_text and isinstance(flat_x_text[0], list):
                        flat_x_text = [t for group in flat_x_text for t in group]
                    labeled_texts = [flat_x_text[i] for i in labeled_indices.tolist()]
                    labeled_embeds = self.module.kb_encoder(
                        labeled_texts, paths=[''] * len(labeled_texts),
                    )

                retrieved = self.kb_manager.search(labeled_embeds, top_k=10)
                id_to_name = self._build_id_to_name_map()
                for cand_list in retrieved:
                    all_candidates_info.append(cand_list)
                    if cand_list:
                        top1 = cand_list[0]
                        pred_id = top1.get("id", "NIL")
                        norm_id = normalize_entity_id(pred_id)
                        pred_name = top1.get("name") or id_to_name.get(norm_id, str(pred_id))
                    else:
                        pred_id = "NIL"
                        pred_name = "NIL"
                    final_pred_ids.append(pred_id)
                    final_pred_names.append(pred_name)
            else:
                final_pred_ids = ["NIL"] * len(labeled_indices)
                final_pred_names = ["NIL"] * len(labeled_indices)
                all_candidates_info = [[] for _ in range(len(labeled_indices))]

            return {
                'final_pred_ids': final_pred_ids,
                'final_pred_names': final_pred_names,
                'target_ent_ids': target_ent_ids,
                'candidates_info': all_candidates_info,
            }

    def get_state_dict(self):
        return {
            'model': self.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
        }

    def load_checkpoint(self, path):
        import time
        max_retries = 5
        for i in range(max_retries):
            try:
                ckpt = torch.load(path, map_location=self.device)
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    missing, unexpected = self.module.load_state_dict(ckpt['model'], strict=False)
                    if ckpt.get('optimizer') is not None:
                        try:
                            self.optimizer.load_state_dict(ckpt['optimizer'])
                            if self.local_rank is None or self.local_rank == 0:
                                print("[Checkpoint] Optimizer state restored.")
                        except Exception as e:
                            if self.local_rank is None or self.local_rank == 0:
                                print(f"[Checkpoint] Optimizer state skipped (param groups changed): {e}")
                    if ckpt.get('scheduler') is not None and hasattr(self, 'scheduler'):
                        try:
                            self.scheduler.load_state_dict(ckpt['scheduler'])
                        except Exception:
                            pass
                else:
                    missing, unexpected = self.module.load_state_dict(ckpt, strict=False)
                if self.local_rank is None or self.local_rank == 0:
                    if missing:
                        print(f"[Checkpoint] Missing keys: {missing}")
                    if unexpected:
                        print(f"[Checkpoint] Unexpected keys (ignored): {unexpected}")
                return
            except (EOFError, RuntimeError) as e:
                if i == max_retries - 1:
                    raise e
                if self.local_rank is None or self.local_rank == 0:
                    print(f"[Checkpoint] Load failed (attempt {i+1}/{max_retries}): {e}, retrying...")
                time.sleep(2)
