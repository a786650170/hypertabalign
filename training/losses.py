import torch
import torch.nn.functional as F

class MultiTaskLoss(torch.nn.Module):
    """
    联合损失函数：检索 + 对齐 + 全局一致性。
    """
    def __init__(self, lambda_cons: float = 0.1):
        super().__init__()
        self.lambda_cons = lambda_cons

    def forward(self, 
                retrieval_logits, retrieval_labels,
                alignment_logits, alignment_labels,
                consistency_data=None):
        
        # 1. 检索损失 (对比学习)
        loss_ret = F.cross_entropy(retrieval_logits, retrieval_labels)
        
        # 2. 对齐损失 (精排)
        loss_align = F.cross_entropy(alignment_logits, alignment_labels)
        
        # 3. 全局一致性损失 (简单实现：同一列预测类别的一致性)
        loss_cons = 0
        if consistency_data is not None:
            # 假设 consistency_data 提供了列索引和预测的类别
            # 这里可以用 KL 散度或方差来约束
            pass
            
        return loss_ret + loss_align + self.lambda_cons * loss_cons

