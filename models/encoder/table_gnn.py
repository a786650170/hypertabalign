import os
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HypergraphConv
from torch_geometric.utils import scatter
from .text_backbone import TextEncoder

# Ablation switch: when set (e.g. "1.0" / "0.0"), forces the text-GNN residual
# gate to a constant value instead of using the learnable sigmoid(gnn_gate_logit).
_FORCE_GATE_ENV = os.environ.get("HYPERTAB_FORCE_GATE", "").strip()
_FORCE_GATE_VAL = float(_FORCE_GATE_ENV) if _FORCE_GATE_ENV else None


class TableGNN(nn.Module):
    """
    Multi-layer Table GNN with disentangled row/column hypergraph convolution
    and learnable text-GNN residual gating.

    Architecture per layer:
        GAT (pairwise local attention) -> residual
        -> Row-HGNN + Col-HGNN (parallel, disentangled) -> gate fusion -> residual
        -> LayerNorm -> Dropout

    After all GNN layers, the output is blended with the original text
    embedding via a learnable gate:  out = (1-alpha)*text + alpha*gnn
    """
    def __init__(self,
                 model_name: str = "microsoft/deberta-v3-small",
                 text_encoder: nn.Module = None,
                 gnn_hidden_dim: int = 768,
                 retrieval_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 order: str = "local_first"):
        super().__init__()
        if text_encoder is not None:
            self.text_encoder = text_encoder
        else:
            self.text_encoder = TextEncoder(model_name)

        if hasattr(self.text_encoder, 'module'):
            in_dim = self.text_encoder.module.hidden_size
        elif hasattr(self.text_encoder, 'hidden_size'):
            in_dim = self.text_encoder.hidden_size
        else:
            in_dim = 768

        self.num_layers = num_layers
        self.order = order
        self.gnn_hidden_dim = gnn_hidden_dim
        self.pre_proj = nn.Linear(in_dim, gnn_hidden_dim)

        self.gnn_gate_logit = nn.Parameter(torch.tensor(-2.0))

        self.edge_type_embed = nn.Embedding(8, gnn_hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATv2Conv(gnn_hidden_dim, gnn_hidden_dim, heads=num_heads, concat=False,
                      edge_dim=gnn_hidden_dim)
            for _ in range(num_layers)
        ])

        self.row_hgnn_layers = nn.ModuleList([
            HypergraphConv(
                gnn_hidden_dim, gnn_hidden_dim,
                use_attention=True, heads=num_heads, concat=False,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.col_hgnn_layers = nn.ModuleList([
            HypergraphConv(
                gnn_hidden_dim, gnn_hidden_dim,
                use_attention=True, heads=num_heads, concat=False,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.rc_gates = nn.ModuleList([
            nn.Linear(gnn_hidden_dim * 2, 2)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(gnn_hidden_dim) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)

        self.extraction_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.alignment_proj = nn.Linear(gnn_hidden_dim, retrieval_dim)

    def _compute_hyperedge_attr(self, x, hyperedge_index):
        return scatter(x[hyperedge_index[0]], hyperedge_index[1], dim=0, reduce='mean')

    def _apply_header_serialization(self, flat_x_text, coords):
        """Prepend column header text to data cell text (Ditto-inspired)."""
        if coords is None or len(coords) == 0:
            return flat_x_text

        coords_list = coords.cpu().tolist() if torch.is_tensor(coords) else coords

        modified = list(flat_x_text)
        col_to_header = {}
        for i, (r, c) in enumerate(coords_list):
            if int(r) == -1:
                col_to_header.setdefault(int(c), modified[i])

        if not col_to_header:
            return modified

        for i, (r, c) in enumerate(coords_list):
            r_int, c_int = int(r), int(c)
            if r_int >= 0 and c_int in col_to_header:
                modified[i] = f"{col_to_header[c_int]} : {modified[i]}"

        return modified

    def forward(self, x_text, edge_index, hyperedge_index=None,
                edge_attr=None, retrieved_context_embeds=None,
                hyperedge_conn_type=None, coords=None):
        if isinstance(x_text, (list, tuple)) and len(x_text) > 0 and isinstance(x_text[0], (list, tuple)):
            flat_x_text = [item for sublist in x_text for item in sublist]
        else:
            flat_x_text = list(x_text) if not isinstance(x_text, list) else x_text

        flat_x_text = self._apply_header_serialization(flat_x_text, coords)

        x = self.text_encoder(flat_x_text)

        if x.dtype != self.pre_proj.weight.dtype:
            x = x.to(dtype=self.pre_proj.weight.dtype)

        x = self.pre_proj(x)
        text_residual = x

        if retrieved_context_embeds is not None:
            if retrieved_context_embeds.shape[-1] == x.shape[-1]:
                x = x + retrieved_context_embeds
        num_nodes = x.size(0)

        edge_embed = None
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            if max_idx >= num_nodes:
                mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, mask]
                if edge_attr is not None:
                    edge_attr = edge_attr[mask]
            if edge_attr is not None:
                edge_embed = self.edge_type_embed(edge_attr.clamp(max=7))

        has_edges = edge_index.numel() > 0
        has_hyperedges = hyperedge_index is not None and hyperedge_index.numel() > 0

        row_he_index = None
        col_he_index = None
        if has_hyperedges and hyperedge_conn_type is not None and hyperedge_conn_type.numel() > 0:
            row_mask = (hyperedge_conn_type == 0)
            col_mask = (hyperedge_conn_type == 1)
            if row_mask.any():
                row_he_index = hyperedge_index[:, row_mask]
            if col_mask.any():
                col_he_index = hyperedge_index[:, col_mask]
        elif has_hyperedges:
            row_he_index = hyperedge_index
            col_he_index = hyperedge_index

        for i in range(self.num_layers):
            x_res = x

            if self.order == "hyper_first":
                x = self._disentangled_hgnn(i, x, row_he_index, col_he_index) + x_res
                if has_edges:
                    x_pre_gat = x
                    x = self.gat_layers[i](x, edge_index, edge_attr=edge_embed)
                    x = torch.relu(x) + x_pre_gat
            else:
                if has_edges:
                    x = self.gat_layers[i](x, edge_index, edge_attr=edge_embed)
                    x = torch.relu(x) + x_res
                if row_he_index is not None or col_he_index is not None:
                    x_pre_hg = x
                    x = self._disentangled_hgnn(i, x, row_he_index, col_he_index) + x_pre_hg

            x = self.norms[i](x)
            if i < self.num_layers - 1:
                x = self.drop(x)

        if _FORCE_GATE_VAL is not None:
            alpha = x.new_tensor(_FORCE_GATE_VAL)
        else:
            alpha = torch.sigmoid(self.gnn_gate_logit)
        x = (1.0 - alpha) * text_residual + alpha * x

        extraction_logits = self.extraction_head(x)
        alignment_embeds = self.alignment_proj(x)

        return extraction_logits, alignment_embeds

    def _disentangled_hgnn(self, layer_idx, x, row_he_index, col_he_index):
        """Run row and column HypergraphConv in parallel, then gate-fuse."""
        if row_he_index is not None:
            row_he_attr = self._compute_hyperedge_attr(x, row_he_index)
            h_row = self.row_hgnn_layers[layer_idx](
                x, row_he_index, hyperedge_attr=row_he_attr
            )
        else:
            h_row = torch.zeros_like(x)

        if col_he_index is not None:
            col_he_attr = self._compute_hyperedge_attr(x, col_he_index)
            h_col = self.col_hgnn_layers[layer_idx](
                x, col_he_index, hyperedge_attr=col_he_attr
            )
        else:
            h_col = torch.zeros_like(x)

        gate = torch.softmax(
            self.rc_gates[layer_idx](torch.cat([h_row, h_col], dim=-1)),
            dim=-1,
        )
        h_fused = gate[:, 0:1] * h_row + gate[:, 1:2] * h_col
        return torch.relu(h_fused)
