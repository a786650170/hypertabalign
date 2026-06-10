import torch
import torch.nn as nn


class RetrievalModule(nn.Module):
    """
    Symmetric retrieval module with shared projection.
    Both query (table cell) and key (KB entity) embeddings pass through
    the same projection to ensure they live in identical matching space.
    """
    def __init__(self, embed_dim: int = 256, proj_dim: int = 256,
                 query_dim: int = None, key_dim: int = None):
        super().__init__()
        actual_dim = embed_dim
        if query_dim is not None:
            actual_dim = query_dim
        self.shared_proj = nn.Linear(actual_dim, proj_dim)
        self.temperature = nn.Parameter(torch.tensor(0.07))
