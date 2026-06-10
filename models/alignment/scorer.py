import torch
import torch.nn as nn

class AlignmentScorer(nn.Module):
    """
    精排与对齐评分模块。
    对检索回来的 Top-K 候选实体进行进一步打分。
    """
    def __init__(self, cell_dim: int = 256, kb_dim: int = 256):
        super().__init__()
        # 交叉特征提取
        self.interaction = nn.Sequential(
            nn.Linear(cell_dim + kb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # 输出标量评分
        )

    def forward(self, cell_embed, cand_embeds):
        """
        cell_embed: [N_cells, dim]
        cand_embeds: [N_cells, K, dim]
        """
        N, K, D = cand_embeds.shape
        # 扩展 cell_embed 以匹配候选数量
        cell_embed_exp = cell_embed.unsqueeze(1).expand(-1, K, -1) # [N, K, D]
        
        # 拼接特征
        features = torch.cat([cell_embed_exp, cand_embeds], dim=-1) # [N, K, 2*D]
        
        # 计算分数
        scores = self.interaction(features).squeeze(-1) # [N, K]
        return scores

