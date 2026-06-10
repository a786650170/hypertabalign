import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from typing import List, Dict, Optional
from models.knowledge.kb_encoder import KBEncoder

class KnowledgeManager:
    """
    仿 HyperGraphRAG 的超图知识管理中心。
    管理 47万+ 实体的向量索引与持久化存储。
    """
    def __init__(self, working_dir: str, model_name: str, device: str = "cuda"):
        self.working_dir = working_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        os.makedirs(working_dir, exist_ok=True)
        # 恢复统一的索引名称
        self.index_path = os.path.join(working_dir, "kb_index.pt")
        self.meta_path = os.path.join(working_dir, "kb_meta.jsonl")
        
        self.embeddings = None
        self.entities = []
        self.is_built = False
        
        # 尝试自动加载已有索引
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load_index()
        else:
            self.is_built = False

    def clear_index(self):
        """彻底删除当前工作目录下的索引产物，避免新旧编码空间混用。"""
        stale_paths = [self.index_path, self.meta_path]
        stale_paths.extend(
            os.path.join(self.working_dir, name)
            for name in os.listdir(self.working_dir)
            if name.startswith("kb_index.pt.part")
        )

        removed = 0
        for path in stale_paths:
            if os.path.exists(path):
                os.remove(path)
                removed += 1

        self.embeddings = None
        self.entities = []
        self.is_built = False
        print(f"[KM] 🧹 已清理旧索引文件 {removed} 个，目录: {self.working_dir}", flush=True)

    def load_index(self):
        try:
            print(f"[KM] 检查索引文件: {self.index_path}", flush=True)
            if not os.path.exists(self.index_path):
                print(f"[KM] ❌ 索引文件不存在: {self.index_path}", flush=True)
                return
            if not os.path.exists(self.meta_path):
                print(f"[KM] ❌ 元数据文件不存在: {self.meta_path}", flush=True)
                return

            print(f"[KM] 📦 正在加载持久化索引: {self.working_dir}...", flush=True)
            self.embeddings = torch.load(self.index_path, map_location="cpu", weights_only=False)
            self.embeddings = F.normalize(self.embeddings, p=2, dim=-1)
            if torch.cuda.is_available():
                cur_dev = torch.cuda.current_device()
                self.embeddings = self.embeddings.to(f"cuda:{cur_dev}", non_blocking=True)
            print(f"[KM] 向量已加载: shape={self.embeddings.shape}, device={self.embeddings.device}", flush=True)

            self.entities = []
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.entities.append(json.loads(line))
            print(f"[KM] 实体元数据已加载: {len(self.entities)} 条", flush=True)

            if len(self.entities) != self.embeddings.size(0):
                print(f"[KM] ⚠️ 数量不匹配 ({len(self.entities)} vs {self.embeddings.size(0)})，索引损坏。", flush=True)
                self.is_built = False
            else:
                self.is_built = True
                print(f"[KM] ✅ 加载完成，共 {len(self.entities)} 个实体 (device={self.embeddings.device}).", flush=True)
        except Exception as e:
            import traceback
            print(f"[KM] ❌ 加载索引失败: {e}", flush=True)
            traceback.print_exc()
            self.is_built = False

    def build_index(
        self,
        kb_path: str,
        encoder: KBEncoder,
        batch_size: int = 1024,
        text_micro_batch_size: int = 64,
        path_chunk_size: int = 512,
        projector: Optional[torch.nn.Module] = None
    ):
        """分布式构建全量索引"""
        
        paths = kb_path
        if isinstance(kb_path, str) and "," in kb_path:
            paths = [p.strip() for p in kb_path.split(",") if p.strip()]
        if isinstance(paths, str):
            paths = [paths]
            
        # DDP Info
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        if rank == 0:
            print(f"🚀 开始构建全量 KB 索引 (分布式模式: {world_size} workers)，源文件: {paths}")
        
        # 1. 读取所有数据 
        # (简单实现：每个进程都读全部元数据，然后只处理自己那部分。对于11M行数据，内存压力可接受)
        raw_data = []
        for path in paths:
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data.append(json.loads(line))
        
        if not raw_data:
            print(f"[Rank {rank}] 没有找到数据，跳过构建。")
            return

        # 2. 数据分片
        my_data = raw_data[rank::world_size]
        print(f"[Rank {rank}] 负责处理 {len(my_data)} / {len(raw_data)} 条数据")
        
        all_embeds = []
        encoder.eval()
        if projector is not None:
            projector.eval()
        
        # 确保 encoder 在正确的 device
        # 如果是 DDP model 的 submodule，它已经在 device 上了
        
        with torch.no_grad():
            iterator = tqdm(range(0, len(my_data), batch_size), desc=f"Rank {rank} Encoding", position=rank)
            for i in iterator:
                batch = my_data[i:i+batch_size]
                names = [e['name'] for e in batch]
                # paths might be missing in some lines
                paths_list = [e.get('path', '') for e in batch]
                
                # 编码
                embeds = encoder(
                    names,
                    paths_list,
                    text_micro_batch_size=text_micro_batch_size,
                    path_chunk_size=path_chunk_size
                ) # [batch, dim]
                if projector is not None:
                    embeds = projector(embeds)
                all_embeds.append(embeds.cpu())
        
        if not all_embeds:
             # Handle empty shard case
             # Need to know the dim. 
             # Hack: run one dummy forward or infer from encoder
             dummy_dim = encoder.fusion.out_features if hasattr(encoder, 'fusion') else 768
             my_embeddings = torch.empty(0, dummy_dim)
        else:
            my_embeddings = torch.cat(all_embeds, dim=0)
            
        # 3. 保存分片结果
        part_path = self.index_path + f".part{rank}"
        torch.save(my_embeddings, part_path)
        print(f"[Rank {rank}] 分片已保存: {part_path}")
        
        # 等待所有进程完成
        if dist.is_initialized():
            dist.barrier()
            
        # 4. Rank 0 合并结果
        if rank == 0:
            print("📦 所有 Rank 完成，正在合并索引...")
            final_embeddings = []
            final_entities = []
            
            # 按 Rank 顺序读取并合并
            # 数据原本是 [0, 1, 2, 3 ...]
            # Rank 0: [0, 4, 8 ...]
            # Rank 1: [1, 5, 9 ...]
            # 合并后: [Rank0..., Rank1...] -> [0, 4, 8..., 1, 5, 9...]
            # 虽然顺序变了，只要 entities 列表也做同样变换，ID 对应关系就是正确的。
            
            for r in range(world_size):
                p = self.index_path + f".part{r}"
                if os.path.exists(p):
                    part_embeds = torch.load(p, map_location="cpu")
                    final_embeddings.append(part_embeds)
                    
                    # 对应的实体
                    part_entities = raw_data[r::world_size]
                    final_entities.extend(part_entities)
                    
                    # 清理临时文件
                    os.remove(p)
            
            if final_embeddings:
                # 构建后直接保存在 CPU，降低后续显存压力
                self.embeddings = torch.cat(final_embeddings, dim=0).cpu()
                self.embeddings = F.normalize(self.embeddings, p=2, dim=-1)
                self.entities = final_entities
                
                # 保存最终结果
                torch.save(self.embeddings, self.index_path)
                with open(self.meta_path, 'w', encoding='utf-8') as f:
                    for e in self.entities:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                        
                self.is_built = True
                print(f"✨ 索引构建成功！总实体数: {len(self.entities)}，路径: {self.working_dir}")
            else:
                print("⚠️ 没有生成任何 Embedding。")

        # 重新加载以保持同步 (对于非 Rank 0)
        if dist.is_initialized():
             dist.barrier()
        
        if rank != 0:
             self.load_index()

    def query(self, query_embeds: torch.Tensor, top_k: int = 10):
        """快速检索接口 (分块版，防止 OOM)"""
        if not self.is_built:
            # 尝试最后一次加载
            self.load_index()
            if not self.is_built:
                print("⚠️ 索引尚未构建，无法查询。返回空结果。")
                return {"scores": [], "indices": [], "entities": []}
            
        # query_embeds: [N, dim]
        query_embeds = F.normalize(query_embeds, p=2, dim=-1)
        num_queries = query_embeds.size(0)
        
        # 结果容器
        all_top_scores = []
        all_top_indices = []
        
        # 采用对 Query 分块的策略，每次处理 128 个 Query
        chunk_size = 128
        
        # 与索引对齐：query 移到 embeddings 所在设备 (默认 CPU)
        target_device = self.embeddings.device
        if query_embeds.device != target_device:
            query_embeds = query_embeds.to(target_device)

        for i in range(0, num_queries, chunk_size):
            q_chunk = query_embeds[i:i+chunk_size] # [chunk, dim]
            
            chunk_scores = torch.matmul(q_chunk, self.embeddings.t()) 
            
            vals, inds = torch.topk(chunk_scores, k=top_k, dim=-1)
            
            all_top_scores.append(vals)
            all_top_indices.append(inds)
            
        # 合并结果
        if not all_top_scores:
             return {"scores": [], "indices": [], "entities": []}
             
        final_scores = torch.cat(all_top_scores, dim=0)
        final_indices = torch.cat(all_top_indices, dim=0)
        
        return {
            "scores": final_scores,
            "indices": final_indices,
            "entities": [[self.entities[idx] for idx in row] for row in final_indices.tolist()]
        }

    def search(self, query_embeds: torch.Tensor, top_k: int = 1):
        """兼容接口: 语义与 query 一致，但直接返回实体列表"""
        res = self.query(query_embeds, top_k)
        return res['entities']
