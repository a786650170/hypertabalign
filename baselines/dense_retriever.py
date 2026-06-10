"""
Shared dense retriever for reranking baselines.
Uses vanilla (pre-trained, NOT fine-tuned) DeBERTa KBEncoder to build a dense index.
No checkpoint is loaded — this ensures fair comparison against the main model.
Caches the index to avoid re-encoding 8M entities every time.
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"
import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from models.encoder.text_backbone import TextEncoder
from models.knowledge.kb_encoder import KBEncoder


class DenseRetriever:
    """Vanilla DeBERTa-based dense retriever with cached KB index. No fine-tuning."""

    def __init__(self, model_name="microsoft/deberta-v3-base",
                 cache_dir=None, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if cache_dir is None:
            cache_dir = os.path.join(PROJECT_ROOT, "experiments/kb_index/dense_retriever_vanilla")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.text_encoder = TextEncoder(model_name, multi_gpu=False)
        self.kb_encoder = KBEncoder(text_encoder=self.text_encoder)
        self.text_encoder.to(self.device)
        self.kb_encoder.to(self.device)
        self.kb_encoder.eval()
        print(f"[DenseRetriever] Using vanilla pre-trained {model_name} (NO checkpoint)")

        self.kb_embeddings = None
        self.kb_entities = None

    def build_or_load_index(self, kb_path):
        """Build KB index or load from cache."""
        index_path = os.path.join(self.cache_dir, "kb_index.pt")
        meta_path = os.path.join(self.cache_dir, "kb_meta.jsonl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            print("[DenseRetriever] Loading cached index...")
            self.kb_embeddings = torch.load(index_path, map_location="cpu")
            self.kb_embeddings = F.normalize(self.kb_embeddings, p=2, dim=-1)
            self.kb_entities = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.kb_entities.append(json.loads(line))
            print(f"  Loaded {len(self.kb_entities)} entities, shape={self.kb_embeddings.shape}")
            return

        print("[DenseRetriever] Building KB index (first time, will be cached)...")
        self.kb_entities = []
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in f:
                self.kb_entities.append(json.loads(line))

        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.kb_entities), 1024), desc="Encoding KB"):
                batch = self.kb_entities[i:i+1024]
                names = [e["name"] for e in batch]
                paths = [e.get("path", "") for e in batch]
                embeds = self.kb_encoder(names, paths,
                                         text_micro_batch_size=64,
                                         path_chunk_size=512)
                all_embeds.append(embeds.cpu())

        self.kb_embeddings = torch.cat(all_embeds, dim=0)
        self.kb_embeddings = F.normalize(self.kb_embeddings, p=2, dim=-1)

        torch.save(self.kb_embeddings, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for e in self.kb_entities:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  Index saved: {self.kb_embeddings.shape}")

    def encode_queries(self, texts, batch_size=64):
        """Encode query texts using the same KBEncoder."""
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                embeds = self.kb_encoder(batch, paths=[""] * len(batch),
                                         text_micro_batch_size=batch_size)
                all_embeds.append(embeds.cpu())
        return F.normalize(torch.cat(all_embeds, dim=0), p=2, dim=-1)

    def retrieve(self, query_embeds, top_k=50):
        """Retrieve top-K candidates for each query. Returns list of list of dicts."""
        results = []
        chunk_size = 256
        for i in range(0, query_embeds.size(0), chunk_size):
            q = query_embeds[i:i+chunk_size]
            scores = torch.matmul(q, self.kb_embeddings.t())
            topk_scores, topk_indices = torch.topk(scores, k=top_k, dim=-1)
            for j in range(q.size(0)):
                cands = []
                for k in range(top_k):
                    idx = topk_indices[j, k].item()
                    cands.append({
                        "id": str(self.kb_entities[idx]["id"]),
                        "name": self.kb_entities[idx]["name"],
                        "score": topk_scores[j, k].item(),
                    })
                results.append(cands)
        return results
