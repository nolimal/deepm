"""Reusable neural-network building blocks for the DeePM architecture.

Contains the temporal backbone (VSN, LSTM, causal attention), graph layers
(dense bipartite GAT / GCN), and supporting modules (ResSwiGLU, ReZero).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

REZERO_DEFAULT = False


# ---------------------------------------------------------------------------
# Feed-forward / gating
# ---------------------------------------------------------------------------

class ResSwiGLU(nn.Module):
    """SwiGLU block with post-norm residual connection.

    Structure: ``x = LayerNorm(x + Dropout(SwiGLU(x)))``
    """

    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.w_gate_val = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.w_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        gate_val = self.w_gate_val(x)
        val, gate = gate_val.chunk(2, dim=-1)
        hidden = val * F.silu(gate)
        hidden = self.dropout1(hidden)
        out = self.w_out(hidden)
        out = self.dropout2(out)
        return self.norm(residual + out)


# ---------------------------------------------------------------------------
# Variable Selection Network (per-variable)
# ---------------------------------------------------------------------------

class PerVariableVSN(nn.Module):
    """Independent processing per feature with FiLM context modulation."""

    def __init__(self, num_vars, hidden_dim, context_dim, dropout=0.2):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        self.feature_embedding = nn.Conv1d(
            in_channels=num_vars,
            out_channels=num_vars * hidden_dim,
            kernel_size=1,
            groups=num_vars,
        )

        self.feature_gate = nn.Conv1d(
            num_vars * hidden_dim, num_vars * hidden_dim, 1, groups=num_vars
        )
        self.feature_val = nn.Conv1d(
            num_vars * hidden_dim, num_vars * hidden_dim, 1, groups=num_vars
        )

        self.film = nn.Linear(context_dim, 2 * (num_vars * hidden_dim))

        nn.init.constant_(self.film.weight, 0.0)
        output_dim = self.film.bias.shape[0]
        midpoint = output_dim // 2
        nn.init.constant_(self.film.bias[:midpoint], 1.0)
        nn.init.constant_(self.film.bias[midpoint:], 0.0)

        self.weight_network = nn.Sequential(
            nn.Linear(num_vars * hidden_dim, num_vars), nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        batch_size, seq_len, _ = x.shape

        x_in = x.permute(0, 2, 1)
        x_embed = self.feature_embedding(x_in)

        flat_input = x_embed.permute(0, 2, 1)

        film_params = self.film(context)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)

        modulated_input = (flat_input * gamma) + beta

        weights = self.weight_network(modulated_input).unsqueeze(-1)

        mod_in_trans = modulated_input.permute(0, 2, 1)

        gate = F.silu(self.feature_gate(mod_in_trans))
        val = self.feature_val(mod_in_trans)

        x_processed = (val * gate).permute(0, 2, 1)
        x_processed = self.dropout(x_processed)
        x_processed = x_processed.view(
            batch_size, seq_len, self.num_vars, self.hidden_dim
        )

        output = (x_processed * weights).sum(dim=2)

        return output, weights


# ---------------------------------------------------------------------------
# Graph layers
# ---------------------------------------------------------------------------

class DenseBipartiteGCN(nn.Module):
    """Dense bipartite GCN layer operating on full adjacency matrices."""

    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, concat=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj, mask=None):
        if isinstance(x, tuple):
            x_source, _ = x
        else:
            x_source = x

        mask_src, mask_tgt = None, None
        if mask is not None:
            if isinstance(mask, tuple):
                mask_src, mask_tgt = mask
            else:
                mask_src = mask_tgt = mask

            if mask_src is not None:
                mask_src = mask_src.bool()
            if mask_tgt is not None:
                mask_tgt = mask_tgt.bool()

        h_src = self.lin(x_source)

        if mask_src is not None:
            h_src = h_src * mask_src.unsqueeze(-1)

        out = torch.bmm(adj, h_src)
        out = out + self.bias

        if mask_tgt is not None:
            out = out * mask_tgt.unsqueeze(-1)

        return out


def masked_softmax(
    scores: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-12
):
    """Softmax that zeros out masked positions; all-masked rows become all-zero."""
    scores = scores.masked_fill(~mask, -1e9)
    attn = F.softmax(scores, dim=dim)
    attn = attn * mask.to(attn.dtype)
    attn = attn / (attn.sum(dim=dim, keepdim=True) + eps)
    return attn


class DenseBipartiteGAT(nn.Module):
    """Dense bipartite Graph Attention layer with multi-head attention."""

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.0, concat=True, use_weights=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self.use_weights = use_weights

        if self.concat:
            self.head_dim = out_channels // heads
            assert (
                self.head_dim * heads == out_channels
            ), "out_channels must be divisible by heads when concat=True"
        else:
            self.head_dim = out_channels

        self.lin_src = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.lin_tgt = nn.Linear(in_channels, heads * self.head_dim, bias=False)

        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_tgt = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_tgt.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_tgt)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj, mask=None):
        if not isinstance(x, tuple):
            raise NotImplementedError("Pass x as (x_source, x_target) for bipartite.")
        x_source, x_target = x

        B, N_src, _ = x_source.shape
        _, N_tgt, _ = x_target.shape
        assert adj.shape[:3] == (B, N_tgt, N_src)

        if mask is None:
            mask_src = None
            mask_tgt = None
        elif isinstance(mask, tuple):
            mask_src, mask_tgt = mask
            mask_src = mask_src.bool()
            mask_tgt = mask_tgt.bool()
        else:
            mask_src = mask.bool()
            mask_tgt = mask.bool()

        h_src = self.lin_src(x_source).view(B, N_src, self.heads, self.head_dim)
        h_tgt = self.lin_tgt(x_target).view(B, N_tgt, self.heads, self.head_dim)

        alpha_src = (h_src * self.att_src).sum(dim=-1)
        alpha_tgt = (h_tgt * self.att_tgt).sum(dim=-1)
        scores = alpha_tgt.unsqueeze(2) + alpha_src.unsqueeze(1)
        scores = F.leaky_relu(scores, negative_slope=0.2)

        if self.use_weights and adj is not None:
            scores = scores + torch.log(adj.unsqueeze(-1) + 1e-9)

        edge_mask = adj != 0
        if mask_tgt is not None:
            edge_mask = edge_mask & mask_tgt.unsqueeze(-1)
        if mask_src is not None:
            edge_mask = edge_mask & mask_src.unsqueeze(-2)

        attn = masked_softmax(scores, edge_mask.unsqueeze(-1), dim=2)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum("btsh,bshd->bthd", attn, h_src)

        if self.concat:
            out = out.reshape(B, N_tgt, self.heads * self.head_dim)
        else:
            out = out.mean(dim=2)

        out = out + self.bias

        if mask_tgt is not None:
            out = out * mask_tgt.unsqueeze(-1).to(out.dtype)

        return out


# ---------------------------------------------------------------------------
# Transformer / attention
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Post-norm Transformer block with optional ReZero gating."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        use_rezero: bool | None = None,
        rezero_init: float = 0.1,
        rezero_learnable: bool = True,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = ResSwiGLU(hidden_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if use_rezero is None:
            should_gate = REZERO_DEFAULT
        else:
            should_gate = use_rezero

        if should_gate:
            self.alpha_attn = nn.Parameter(torch.tensor(float(rezero_init)))
            self.alpha_attn.requires_grad = rezero_learnable
        else:
            self.register_buffer("alpha_attn", torch.tensor(1.0))

    def forward(self, x, key=None, value=None, mask=None, key_padding_mask=None):
        residual = x

        k = key if key is not None else x
        v = value if value is not None else x

        attn_out, _ = self.attn(
            query=x,
            key=k,
            value=v,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )

        x = self.norm1(residual + self.alpha_attn * self.dropout(attn_out))
        x = self.ffn(x)

        return x


# ---------------------------------------------------------------------------
# ReZero wrapper & GNN block
# ---------------------------------------------------------------------------

class ReZeroWrapper(nn.Module):
    """Wraps any layer with a learnable residual scale alpha."""

    def __init__(self, gnn_layer: nn.Module, alpha_init: float = 0.0, learnable: bool = True):
        super().__init__()
        self.gnn = gnn_layer
        alpha_t = torch.tensor(float(alpha_init))
        if learnable:
            self.alpha = nn.Parameter(alpha_t)
        else:
            self.register_buffer("alpha", alpha_t)

    def forward(self, *args, **kwargs):
        out = self.gnn(*args, **kwargs)
        return out * self.alpha


class GNNBlock(nn.Module):
    """Post-norm GNN block: ``x = FFN(LayerNorm(x + Dropout(GNN(x_source))))``."""

    def __init__(
        self, hidden_dim: int, dropout: float, gnn_type: str = "gat", heads: int = 4,
        use_rezero: bool | None = None,
        rezero_init: float = 0.1,
        rezero_learnable: bool = True,
        gat_concat_heads: bool = False,
        gnn_weighted_edges: bool = False,
    ):
        super().__init__()

        if gnn_type == "gat":
            base_gnn = DenseBipartiteGAT(
                hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=gat_concat_heads, use_weights=gnn_weighted_edges
            )
        elif gnn_type == "gcn":
            base_gnn = DenseBipartiteGCN(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        if use_rezero is None:
            use_rezero = REZERO_DEFAULT

        if use_rezero:
            self.gnn = ReZeroWrapper(base_gnn, alpha_init=rezero_init, learnable=rezero_learnable)
        else:
            self.gnn = base_gnn
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = ResSwiGLU(hidden_dim, dropout=dropout)

    def forward(self, x_target, adj, x_source=None, mask=None):
        if x_source is None:
            x_source = x_target

        mask_src = mask_tgt = None
        if mask is not None:
            if isinstance(mask, tuple):
                mask_src, mask_tgt = mask
            else:
                mask_src = mask_tgt = mask
            if mask_src is not None:
                mask_src = mask_src.bool()
            if mask_tgt is not None:
                mask_tgt = mask_tgt.bool()

        residual = x_target
        if mask_tgt is not None:
            residual = residual * mask_tgt.unsqueeze(-1)

        if (mask_src is not None) or (mask_tgt is not None):
            gnn_out = self.gnn((x_source, x_target), adj, mask=(mask_src, mask_tgt))
        else:
            gnn_out = self.gnn((x_source, x_target), adj)

        x = self.norm(residual + self.dropout(gnn_out))
        x = self.ffn(x)

        if mask_tgt is not None:
            x = x * mask_tgt.unsqueeze(-1)

        return x


# ---------------------------------------------------------------------------
# Temporal backbone
# ---------------------------------------------------------------------------

class TemporalBackbone(nn.Module):
    """LSTM-based temporal encoder with per-variable VSN and optional correlation fusion."""

    def __init__(
        self,
        num_tickers,
        num_vars,
        hidden_dim,
        num_heads,
        dropout=0.1,
        fuse_correlation_input=False,
        corr_input_dim=0,
    ):
        super().__init__()
        self.fuse_correlation = fuse_correlation_input

        self.ticker_embedding = nn.Embedding(num_tickers, hidden_dim)
        self.cost_to_ctx = nn.Linear(hidden_dim + 1, hidden_dim)

        if self.fuse_correlation:
            self.corr_projection = nn.Linear(corr_input_dim, hidden_dim)
            self.fusion_block = ResSwiGLU(hidden_dim, dropout=dropout)

        self.vsn = PerVariableVSN(num_vars, hidden_dim, hidden_dim, dropout)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.lstm_gate = ResSwiGLU(hidden_dim, dropout=dropout)

        self.temporal_attn = TransformerBlock(hidden_dim, num_heads, dropout)

        self.init_h = nn.Linear(hidden_dim, hidden_dim)
        self.init_c = nn.Linear(hidden_dim, hidden_dim)

    def _generate_causal_mask(self, size, device):
        """Generates upper-triangular mask to prevent future leakage."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        sequence,
        ticker_idx,
        correlation_features=None,
        cost_context=None,
    ):
        static_context = self.ticker_embedding(ticker_idx)

        if cost_context is not None:
            static_context = self.cost_to_ctx(
                torch.cat([static_context, cost_context], dim=-1)
            )

        x, importance_weights = self.vsn(sequence, static_context)

        if self.fuse_correlation and correlation_features is not None:
            corr_proj = self.corr_projection(correlation_features)
            x = self.fusion_block(x + corr_proj)

        h_0 = torch.tanh(self.init_h(static_context)).unsqueeze(0)
        c_0 = torch.tanh(self.init_c(static_context)).unsqueeze(0)

        lstm_out, _ = self.lstm(x, (h_0, c_0))

        x_adapted = self.lstm_gate(lstm_out)

        seq_len = x_adapted.shape[1]
        causal_mask = self._generate_causal_mask(seq_len, x_adapted.device)

        x_temporal = self.temporal_attn(x_adapted, mask=causal_mask)

        return x_temporal, importance_weights
