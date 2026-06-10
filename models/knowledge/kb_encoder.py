import torch
import torch.nn as nn
from ..encoder.text_backbone import TextEncoder

class TaxonomyEncoder(nn.Module):
    """
    专门针对 Schema.org 等层级体系的编码器。
    它不仅处理文本，还感知层级的深度与父子关系。
    """
    def __init__(self, text_encoder, embed_dim=256):
        super().__init__()
        self.text_encoder = text_encoder
        self.embed_dim = embed_dim
        self.level_embed = nn.Embedding(10, embed_dim)

        hidden_size = getattr(text_encoder, 'module', text_encoder).hidden_size
        self.gru = nn.GRU(hidden_size, embed_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, path_list: list[str], text_micro_batch_size: int = None, chunk_size: int = 64):
        all_segments = []
        segment_counts = []

        for path in path_list:
            if not path:
                segments = ["Unknown"]
            else:
                segments = [s.strip() for s in path.split(">")]
            all_segments.extend(segments)
            segment_counts.append(len(segments))

        all_seg_embeds_list = []

        if text_micro_batch_size is None:
            text_micro_batch_size = 8

        for i in range(0, len(all_segments), chunk_size):
            batch_segs = all_segments[i:i+chunk_size]
            emb = self.text_encoder(batch_segs, micro_batch_size=text_micro_batch_size)
            all_seg_embeds_list.append(emb)

        all_seg_embeds = torch.cat(all_seg_embeds_list, dim=0)

        if all_seg_embeds.dtype != self.proj.weight.dtype:
            all_seg_embeds = all_seg_embeds.to(dtype=self.proj.weight.dtype)

        batch_size = len(path_list)
        max_len = max(segment_counts)
        hidden_dim = all_seg_embeds.size(1)

        padded_input = torch.zeros(
            batch_size, max_len, hidden_dim,
            device=all_seg_embeds.device, dtype=all_seg_embeds.dtype,
        )
        current_idx = 0
        for i, count in enumerate(segment_counts):
            padded_input[i, :count, :] = all_seg_embeds[current_idx:current_idx + count]
            current_idx += count

        gru_out, _ = self.gru(padded_input)

        last_indices = torch.tensor(segment_counts, device=gru_out.device) - 1
        batch_indices = torch.arange(batch_size, device=gru_out.device)
        last_hidden = gru_out[batch_indices, last_indices, :]

        path_repr = self.proj(last_hidden)

        depth = last_indices.clamp(max=9)
        path_repr = path_repr + self.level_embed(depth)

        return path_repr

class KBEncoder(nn.Module):
    """
    知识库实体编码器 (Schema.org 增强版)。
    联合编码实体名称及其在本体论中的层级位置。
    """
    def __init__(self, 
                 model_name: str = "microsoft/deberta-v3-small",
                 text_encoder: nn.Module = None,
                 kb_embed_dim: int = 256):
        super().__init__()
        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            self.text_encoder = TextEncoder(model_name)
        self.taxonomy_encoder = TaxonomyEncoder(self.text_encoder, kb_embed_dim)
        
        # 兼容 DataParallel 包装
        hidden_size = getattr(self.text_encoder, 'module', self.text_encoder).hidden_size
        
        # 增加一个专用的名称投影层，替代之前的粗暴裁剪
        self.name_proj = nn.Linear(hidden_size, kb_embed_dim)
        
        # 最终融合层：将实体名向量与路径拓扑向量融合
        self.fusion = nn.Linear(kb_embed_dim + kb_embed_dim, kb_embed_dim)

    def forward(
        self,
        names: list[str],
        paths: list[str] = None,
        text_micro_batch_size: int = None,
        path_chunk_size: int = 128,
        return_name_features: bool = False,
    ):
        name_embeds = self.text_encoder(names, micro_batch_size=text_micro_batch_size)

        if name_embeds.dtype != self.name_proj.weight.dtype:
            name_embeds = name_embeds.to(dtype=self.name_proj.weight.dtype)

        name_features = self.name_proj(name_embeds)

        unique_paths = set(p for p in paths if p) if paths else set()
        use_paths = len(unique_paths) > 1

        if use_paths:
            path_features = self.taxonomy_encoder(
                paths,
                text_micro_batch_size=text_micro_batch_size,
                chunk_size=path_chunk_size
            )
            combined = torch.cat([name_features, path_features], dim=-1)
            out = self.fusion(combined)
        else:
            out = name_features

        if return_name_features:
            return out, name_features
        return out

