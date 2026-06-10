import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder.text_backbone import TextEncoder
from models.encoder.table_gnn import TableGNN
from models.knowledge.kb_encoder import KBEncoder
from models.retrieval.dual_encoder import RetrievalModule


class HyperGraphRAGModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_name = config.get('model_name', 'microsoft/deberta-v3-small')
        gnn_order = config.get('gnn_order', 'local_first')

        gnn_layers = int(config.get('gnn_layers', 2))
        gnn_dropout = float(config.get('gnn_dropout', 0.1))

        gnn_hidden_dim = int(config.get('gnn_hidden_dim', 768))
        retrieval_dim = int(config.get('retrieval_dim', 256))

        self.shared_text_encoder = TextEncoder(model_name, multi_gpu=False)
        self.table_encoder = TableGNN(
            text_encoder=self.shared_text_encoder,
            gnn_hidden_dim=gnn_hidden_dim,
            retrieval_dim=retrieval_dim,
            num_layers=gnn_layers,
            dropout=gnn_dropout,
            order=gnn_order,
        )
        self.kb_encoder = KBEncoder(
            text_encoder=self.shared_text_encoder,
            kb_embed_dim=retrieval_dim,
        )
        self.retriever = RetrievalModule(
            embed_dim=retrieval_dim,
            proj_dim=retrieval_dim,
        )

    def _table_encoder_kwargs(self, graph, **extra):
        return dict(
            hyperedge_index=getattr(graph, 'hyperedge_index', None),
            edge_attr=getattr(graph, 'edge_attr', None),
            hyperedge_conn_type=getattr(graph, 'hyperedge_conn_type', None),
            coords=getattr(graph, 'coords', None),
            **extra,
        )

    def forward(self, graph, mode='train', **kwargs):
        if mode == 'encode_table':
            return self.table_encoder(
                graph.x_text,
                graph.edge_index,
                **self._table_encoder_kwargs(
                    graph,
                    retrieved_context_embeds=kwargs.get('retrieved_context_embeds'),
                ),
            )

        elif mode == 'encode_kb':
            return self.kb_encoder(kwargs['names'], kwargs.get('paths', None))

        elif mode == 'train_forward':
            extraction_logits, alignment_embeds = self.table_encoder(
                graph.x_text,
                graph.edge_index,
                **self._table_encoder_kwargs(graph),
            )

            valid_cell_indices = kwargs.get('valid_cell_indices')
            if valid_cell_indices is None or len(valid_cell_indices) == 0:
                return extraction_logits, None, None, None, None

            labeled_cell_embeds = alignment_embeds[valid_cell_indices]

            target_names = kwargs['target_names']
            target_paths = kwargs.get('target_paths', [''] * len(target_names))

            target_kb_embeds, name_embeds_proj = self.kb_encoder(
                target_names, target_paths, return_name_features=True,
            )

            query_features = F.normalize(
                self.retriever.shared_proj(labeled_cell_embeds), p=2, dim=-1
            )
            key_features = F.normalize(
                self.retriever.shared_proj(target_kb_embeds), p=2, dim=-1
            )
            retrieval_logits = torch.matmul(
                query_features, key_features.t()
            ) / self.retriever.temperature

            neg_names = kwargs.get('neg_names')
            neg_paths = kwargs.get('neg_paths')
            neg_logits = None
            if neg_names and len(neg_names) > 0:
                neg_kb_embeds = self.kb_encoder(neg_names, neg_paths)
                neg_key_features = F.normalize(
                    self.retriever.shared_proj(neg_kb_embeds), p=2, dim=-1
                )
                neg_logits = torch.matmul(
                    query_features, neg_key_features.t()
                ) / self.retriever.temperature

            return (
                extraction_logits,
                retrieval_logits,
                neg_logits,
                labeled_cell_embeds,
                name_embeds_proj,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")
