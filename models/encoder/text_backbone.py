import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    """
    使用 HuggingFace Transformer 模型编码文本。
    """
    def __init__(self, model_name: str = "microsoft/deberta-v3-small", multi_gpu: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size

        # 针对 H100 重新开启梯度检查点，并使用非重入模式解决共享报错
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            print(f"🚀 [Memory Opt] 开启 {model_name} 梯度检查点 (Non-reentrant)...")
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # 多 GPU 优化：在内部对 backbone 使用 DataParallel
        # ⚠️ CRITICAL FIX: 在 DDP 模式下，严禁开启内部 DataParallel
        # 如果当前进程已经有 local_rank 环境变量，说明在 DDP 中，强制关闭 internal multi_gpu
        import os
        if "LOCAL_RANK" in os.environ:
            multi_gpu = False
            
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"检测到 {torch.cuda.device_count()} 张显卡，已在 TextEncoder 内部开启多卡并行。")
            self.backbone = nn.DataParallel(self.backbone)

    def forward(self, texts: list[str], micro_batch_size: int = None):
        """
        分批编码文本。由于开启了梯度检查点，显存占用将大幅下降。
        """
        if not texts:
            # 获取 device，兼容 DataParallel
            if isinstance(self.backbone, nn.DataParallel):
                device = self.backbone.module.device
            else:
                device = self.backbone.device
            # 实际上直接用 list parameters 更稳妥
            try:
                device = next(self.backbone.parameters()).device
            except:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            return torch.zeros((0, self.hidden_size)).to(device)
            
        texts = [str(t) if t is not None else "" for t in texts]
        try:
            device = next(self.backbone.parameters()).device
        except: # DataParallel 有时取不到 parameters
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 动态调整 micro_batch_size
        if micro_batch_size is None:
            # DDP 模式下，每个进程只看到一张卡，所以 isinstance(DP) 为 False
            # 此时 micro_batch_size 控制的是**单张卡**一次塞给 TextEncoder 的句子数
            # 虽然我们把总 batch size 设为 64，但 TextEncoder 处理的是 graph.x_text (全部展平)
            # 一个表格可能有几百个单元格，展平后可能是几百个句子
            # 必须设置一个较小的 micro_batch 防止单次 forward 爆炸
            
            micro_batch_size = 32
        
        all_embeds = []
        for i in range(0, len(texts), micro_batch_size):
            batch_texts = texts[i:i+micro_batch_size]
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(device)
            
            outputs = self.backbone(**inputs)
            # DataParallel 包装后的 outputs 结构不变
            pooler_output = outputs.last_hidden_state[:, 0, :]
            
            # 强制转换为 float32，避免后续计算出现 dtype 不匹配或溢出 NaN
            pooler_output = pooler_output.to(dtype=torch.float32)
            
            # 仅仅检测并打印，不修改数据 (保持学术严谨性，交给 Trainer 跳过该 Batch)
            if torch.isnan(pooler_output).any():
                print(f"⚠️ [TextEncoder] NaNs detected in batch {i}!")
                
            all_embeds.append(pooler_output)
            
        return torch.cat(all_embeds, dim=0)

