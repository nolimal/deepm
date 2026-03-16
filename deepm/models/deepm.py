import math

import torch
import torch.nn as nn
import pandas as pd

from deepm._paths import PROJECT_ROOT
from deepm.models.base import DeepMomentumNetwork
from deepm.models.deepm_layers import (
    GNNBlock,
    ResSwiGLU,
    TemporalBackbone,
    TransformerBlock,
)


# --- MAIN MODEL ---


class SpatioTemporalGraphTransformer(DeepMomentumNetwork):
    """Main DeePM model: spatio-temporal graph transformer for cross-sectional momentum.

    Combines a per-asset temporal backbone (VSN -> LSTM -> Temporal Attention)
    with cross-asset interaction via bipartite graph attention (GAT/GCN) and/or
    cross-attention. Trained end-to-end with a differentiable Sharpe ratio loss.

    Key architectural features:
        - Variable Selection Network (VSN) for adaptive feature weighting
        - LSTM for temporal sequence encoding
        - Causal temporal self-attention for refinement
        - Optional bipartite GNN (GAT or GCN) for cross-asset information flow
        - Optional cross-attention between assets
        - Position sizing via a learned output head
    """

    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        # GNN Args
        use_xatt: bool = True,
        use_gnn: bool = False,
        gnn_type: str = "gat",
        adjacency_file: str = str(PROJECT_ROOT / "macro_adjacency_normalized.csv"),
        expected_ticker_order: list = None,
        # Architecture Logic
        use_lagged_cross_section: bool = True,
        gnn_first: bool = False,
        # Correlation / Fusion Args
        use_correlation_features: bool = False,
        correlation_feature_location: int = 0,  # 0=Input, 1=Output, 2=Both
        use_rezero_attn: bool | None = None,
        use_rezero_gnn: bool | None = None,
        rezero_init: float = 0.1,
        rezero_learnable: bool = True,
        gat_concat_heads: bool = False,
        gnn_weighted_edges: bool = False,
        **kwargs,
    ):
        use_staggered_close_kv = kwargs.get("use_staggered_close_kv", False)
        ticker_close_rank = kwargs.get("ticker_close_rank", None)

        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs,
        )

        self.gnn_weighted_edges = gnn_weighted_edges
        
        use_single_head_spatial = kwargs.get("use_single_head_spatial", False)
        spatial_num_heads = 1 if use_single_head_spatial else num_heads

        self.num_tickers = num_tickers
        self.hidden_dim = hidden_dim
        self.seq_len = kwargs["seq_len"]

        self.use_xatt = use_xatt
        self.use_gnn = use_gnn
        self.use_lagged_cross_section = use_lagged_cross_section
        self.gnn_first = gnn_first

        self.use_staggered_close_kv = use_staggered_close_kv
        if self.use_staggered_close_kv and not self.use_lagged_cross_section:
            raise ValueError(
                "use_staggered_close_kv=True requires use_lagged_cross_section=True "
                "(we need a previous-day representation)."
            )

        if self.use_staggered_close_kv:
            if ticker_close_rank is None:
                raise ValueError(
                    "use_staggered_close_kv=True but ticker_close_rank is None. "
                    "Pass dataset.ticker_close_rank via TrainDeepMomentumNetwork._load_architecture."
                )
            close_rank_tensor = torch.as_tensor(
                ticker_close_rank, dtype=torch.long, device=self.device
            )
            if close_rank_tensor.ndim != 1 or close_rank_tensor.numel() != num_tickers:
                raise ValueError(
                    f"ticker_close_rank must be 1D with length num_tickers={num_tickers}, "
                    f"got shape {tuple(close_rank_tensor.shape)}."
                )
            self.register_buffer("ticker_close_rank", close_rank_tensor)
            self.num_close_groups = int(close_rank_tensor.max().item()) + 1
        else:
            # Placeholder buffer for uniform code paths
            close_rank_tensor = torch.arange(
                num_tickers, dtype=torch.long, device=self.device
            )
            self.register_buffer("ticker_close_rank", close_rank_tensor)
            self.num_close_groups = 1

        # --- Correlation Config ---
        self.use_corr = use_correlation_features
        self.corr_loc = correlation_feature_location
        self.corr_dim = num_tickers

        fuse_at_input = self.use_corr and (self.corr_loc in [0, 2])
        self.fuse_at_output = self.use_corr and (self.corr_loc in [1, 2])

        if self.fuse_at_output:
            self.output_fusion_block = ResSwiGLU(hidden_dim, dropout=dropout)
            self.corr_out_proj = nn.Linear(self.corr_dim, hidden_dim)

        # --- 1. Temporal backbone ---
        # base dynamic features + optional vs_factor + mask_single_date
        seq_input_dim = self.input_dim + getattr(self, "extra_tcost_channels", 0) + 1

        self.seq_rep = TemporalBackbone(
            num_tickers=num_tickers,
            num_vars=seq_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            fuse_correlation_input=fuse_at_input,
            corr_input_dim=self.corr_dim,
        )

        # --- 2. Cross-sectional attention ---
        if self.use_xatt:
            self.cross_att = TransformerBlock(
                hidden_dim, 
                spatial_num_heads, 
                dropout,
                use_rezero=use_rezero_attn,
                rezero_init=rezero_init,
                rezero_learnable=rezero_learnable,
            )

        # --- 3. Graph Neural Network (spatial) ---
        if self.use_gnn:
            self.gnn_block = GNNBlock(
                hidden_dim, dropout, gnn_type, spatial_num_heads,
                use_rezero=use_rezero_gnn,
                rezero_init=rezero_init,
                rezero_learnable=rezero_learnable,
                gat_concat_heads=gat_concat_heads,
                gnn_weighted_edges=gnn_weighted_edges,
            )

            self._init_adjacency(adjacency_file, expected_ticker_order, gnn_type)

        self.is_cross_section = True

    def _init_adjacency(self, file_path, ticker_order, gnn_type):
        """Simple dense adjacency loader."""
        if not file_path:
            raise ValueError("Adjacency file required for GNN")
        df = pd.read_csv(file_path, index_col=0)

        if ticker_order:
            df = df.reindex(index=ticker_order, columns=ticker_order).fillna(0.0)
        else:
            df = df.sort_index(axis=0).sort_index(axis=1)

        adj_tensor = torch.tensor(df.values, dtype=torch.float32)
        if gnn_type == "gat":
            if not self.gnn_weighted_edges:
                adj_tensor = (adj_tensor > 0).float()
            else:
                adj_tensor = torch.relu(adj_tensor)
        self.register_buffer("adj_matrix", adj_tensor)

    def _shift_tensor(self, x):
        """
        Shift along the time dimension by 1 (previous-day cross-section).

        x: [B, S, N, H]
        """
        shifted = torch.zeros_like(x)
        shifted[:, 1:, :] = x[:, :-1, :]
        return shifted

    def _staggered_close_cross_att(
        self,
        x_query: torch.Tensor,
        x_same: torch.Tensor,
        x_prev: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-sectional attention with close-time–aware K/V.

        Inputs:
            x_query: [B*S, N, H]  -- queries for each ticker i at its own close time
            x_same:  [B*S, N, H]  -- same-day representation
            x_prev:  [B*S, N, H]  -- previous-day representation
            key_padding_mask: [B*S, N]  -- True = pad (invalid ticker)

        For a ticker i in close group g_i, K/V for ticker j is:
            K_j, V_j = x_same_j if group_j <= g_i else x_prev_j.

        We implement this by looping over close groups g and processing
        all tickers with group == g together.
        """
        # If staggered logic disabled or trivial, fall back to lag-all behaviour.
        if (not self.use_staggered_close_kv) or (self.num_close_groups <= 1):
            return self.cross_att(
                x=x_query,
                key=x_prev,
                value=x_prev,
                key_padding_mask=key_padding_mask,
            )

        BSt, N, H = x_query.shape
        close_rank = self.ticker_close_rank  # [N]

        x_out = torch.zeros_like(x_query)

        for g in range(self.num_close_groups):
            group_mask = close_rank == g
            if not torch.any(group_mask):
                continue

            # same-day for tickers with close_rank <= g, previous-day otherwise
            same_mask = (close_rank <= g).float().view(1, N, 1)  # [1, N, 1]
            kv = same_mask * x_same + (1.0 - same_mask) * x_prev  # [BSt, N, H]

            # Queries only for tickers in this group
            Q = x_query[:, group_mask, :]  # [BSt, N_g, H]

            updated = self.cross_att(
                x=Q,
                key=kv,
                value=kv,
                key_padding_mask=key_padding_mask,
            )
            x_out[:, group_mask, :] = updated

        return x_out

    def _staggered_close_gnn(
        self,
        x_target: torch.Tensor,   # [B*S, N, H]
        x_same: torch.Tensor,     # [B*S, N, H]  (same-day states to use when not future)
        x_prev: torch.Tensor,     # [B*S, N, H]  (previous-day states to use when future)
        adj: torch.Tensor,        # [B*S, N, N]  (full adjacency)
        gnn_mask: torch.Tensor,   # [B*S, N]     (True = valid)
    ) -> torch.Tensor:

        # If disabled/trivial, fall back to "all prev-day source"
        if (not self.use_staggered_close_kv) or (self.num_close_groups <= 1):
            return self.gnn_block(
                x_target=x_target,
                adj=adj,
                x_source=x_prev,
                mask=gnn_mask,
            )

        BSt, N, H = x_target.shape
        close_rank = self.ticker_close_rank  # [N]

        x_out = torch.zeros_like(x_target)

        for g in range(self.num_close_groups):
            group_mask = (close_rank == g)
            if not torch.any(group_mask):
                continue

            # Source gating: same-day if not future, else previous-day
            same_mask = (close_rank <= g).float().view(1, N, 1)     # [1, N, 1]
            src = same_mask * x_same + (1.0 - same_mask) * x_prev   # [BSt, N, H]

            # Update only targets in this close group
            tgt = x_target[:, group_mask, :]        # [BSt, N_g, H]
            adj_sub = adj[:, group_mask, :]         # [BSt, N_g, N]

            # Bipartite masks: (src mask over N, tgt mask over N_g)
            mask_src = gnn_mask
            mask_tgt = gnn_mask[:, group_mask]

            updated = self.gnn_block(
                x_target=tgt,
                adj=adj_sub,
                x_source=src,
                mask=(mask_src, mask_tgt),
            )

            x_out[:, group_mask, :] = updated

        return x_out


    def variable_importance(
        self, target_x, target_tickers, mask_single_date, batch_size, **kwargs
    ):
        """Extract variable selection weights for interpretation."""
        x_in = torch.cat([target_x, mask_single_date.float().unsqueeze(-1)], dim=-1)

        x_in = x_in.view(batch_size * self.num_tickers, self.seq_len, -1)

        tickers_flat = target_tickers.view(-1)
        static_context = self.seq_rep.ticker_embedding(tickers_flat)

        _, weights = self.seq_rep.vsn(x_in, static_context)

        weights = weights.squeeze(-1)  # [B*N, S, Num_Vars]
        weights = weights.view(batch_size, self.num_tickers, self.seq_len, -1)

        return weights

    def forward_candidate_arch(
        self,
        target_x,
        target_tickers,
        mask_entire_sequence,
        mask_single_date,
        batch_size,
        corr_features=None,
        **kwargs,
    ):
        # --- A. Temporal phase ---
        trans_cost_bp = kwargs.get("trans_cost_bp", None)

        x_in = torch.cat([target_x, mask_single_date.float().unsqueeze(-1)], dim=-1)
        x_in = x_in.view(batch_size * self.num_tickers, self.seq_len, -1)

        tickers_flat = target_tickers.view(-1)

        corr_in = None
        if self.use_corr and corr_features is not None:
            corr_in = corr_features.view(
                batch_size * self.num_tickers, self.seq_len, -1
            )

        # Build static cost context: one value per (batch, asset)
        cost_context = None
        if trans_cost_bp is not None and self.use_transaction_costs:
            if self.logscale_cost_inputs:
                cost_static = (
                    trans_cost_bp[:, :, 0] * self.trans_cost_scaler
                ).log1p() / math.log(
                    2
                )  # [B, N]
            else:
                cost_static = trans_cost_bp[:, :, 0] * self.trans_cost_scaler
            cost_context = cost_static.reshape(batch_size * self.num_tickers, 1)

        x_rep, _ = self.seq_rep(
            x_in,
            tickers_flat,
            correlation_features=corr_in,
            cost_context=cost_context,
        )

        # --- B. Reshape to [B, S, N, H] & build lagged version ---
        x_struct = x_rep.view(
            batch_size, self.num_tickers, self.seq_len, self.hidden_dim
        )
        x_struct = x_struct.permute(0, 2, 1, 3).contiguous()  # [B, S, N, H]
        x_cross = x_struct.view(-1, self.num_tickers, self.hidden_dim)  # [B*S, N, H]

        if self.use_lagged_cross_section:
            x_shifted = self._shift_tensor(x_struct)  # [B, S, N, H] (t-1)
            x_lagged = x_shifted.view(-1, self.num_tickers, self.hidden_dim)
        else:
            x_lagged = x_cross

        # --- C. Masks ---
        # mask_entire_sequence: [B, N] (True = never-observed ticker)
        seq_mask = mask_entire_sequence.unsqueeze(1).expand(
            -1, self.seq_len, -1
        )  # [B, S, N]
        flat_mask = seq_mask.reshape(batch_size * self.seq_len, self.num_tickers)
        cross_attn_padding_mask = flat_mask  # True = pad / invalid
        gnn_mask = (~flat_mask).to(torch.bool)  # True = valid node for GNN

        # --- D. Spatial processing order ---
        if self.gnn_first:
            # 1) GNN first
            if self.use_gnn:
                batch_adj = self.adj_matrix.unsqueeze(0).expand(x_cross.size(0), -1, -1)
            if self.use_staggered_close_kv:
                x_cross = self._staggered_close_gnn(
                    x_target=x_cross,
                    x_same=x_cross,      # same-day states (pre-GNN)
                    x_prev=x_lagged,     # t-1 states
                    adj=batch_adj,
                    gnn_mask=gnn_mask,
                )
            else:
                x_cross = self.gnn_block(
                    x_target=x_cross,
                    adj=batch_adj,
                    x_source=x_lagged,
                    mask=gnn_mask,
                )
                x_cross = x_cross.masked_fill(flat_mask.unsqueeze(-1), 0.0)

            # 2) Cross-sectional attention (on post-GNN representation)
            if self.use_xatt:
                if self.use_staggered_close_kv:
                    if self.use_lagged_cross_section:
                        x_struct_after_gnn = x_cross.view(
                            batch_size, self.seq_len, self.num_tickers, self.hidden_dim
                        )
                        x_shifted_after_gnn = self._shift_tensor(x_struct_after_gnn)
                        x_prev = x_shifted_after_gnn.view(
                            -1, self.num_tickers, self.hidden_dim
                        )
                    else:
                        x_prev = x_cross

                    x_cross = self._staggered_close_cross_att(
                        x_query=x_cross,
                        x_same=x_cross,
                        x_prev=x_prev,
                        key_padding_mask=cross_attn_padding_mask,
                    )
                else:
                    if self.use_lagged_cross_section:
                        x_struct_after_gnn = x_cross.view(
                            batch_size,
                            self.seq_len,
                            self.num_tickers,
                            self.hidden_dim,
                        )
                        x_shifted_after_gnn = self._shift_tensor(x_struct_after_gnn)
                        attn_src = x_shifted_after_gnn.view(
                            -1, self.num_tickers, self.hidden_dim
                        )
                    else:
                        attn_src = x_cross

                    x_cross = self.cross_att(
                        x=x_cross,
                        key=attn_src,
                        value=attn_src,
                        key_padding_mask=cross_attn_padding_mask,
                    )

                x_cross = x_cross.masked_fill(flat_mask.unsqueeze(-1), 0.0)

        else:
            # --- Order: Attention -> GNN ---
            # 1) Cross-sectional attention first
            if self.use_xatt:
                if self.use_staggered_close_kv:
                    x_cross = self._staggered_close_cross_att(
                        x_query=x_cross,
                        x_same=x_cross,
                        x_prev=x_lagged,
                        key_padding_mask=cross_attn_padding_mask,
                    )
                else:
                    x_cross = self.cross_att(
                        x=x_cross,
                        key=x_lagged,
                        value=x_lagged,
                        key_padding_mask=cross_attn_padding_mask,
                    )

                x_cross = x_cross.masked_fill(flat_mask.unsqueeze(-1), 0.0)

            # 2) GNN second
            if self.use_gnn:
                if self.use_lagged_cross_section:
                    x_struct_after_attn = x_cross.view(
                        batch_size, self.seq_len, self.num_tickers, self.hidden_dim
                    )
                    x_shifted_after_attn = self._shift_tensor(x_struct_after_attn)
                    gnn_src = x_shifted_after_attn.view(
                        -1, self.num_tickers, self.hidden_dim
                    )
                else:
                    gnn_src = x_cross

                batch_adj = self.adj_matrix.unsqueeze(0).expand(x_cross.size(0), -1, -1)
                if self.use_staggered_close_kv:
                    x_cross = self._staggered_close_gnn(
                        x_target=x_cross,
                        x_same=x_cross,   # same-day post-attn state
                        x_prev=gnn_src,   # shifted (t-1) post-attn state
                        adj=batch_adj,
                        gnn_mask=gnn_mask,
                    )
                else:
                    x_cross = self.gnn_block(
                        x_target=x_cross,
                        adj=batch_adj,
                        x_source=gnn_src,
                        mask=gnn_mask,
                    )

                x_cross = x_cross.masked_fill(flat_mask.unsqueeze(-1), 0.0)

        # --- E. Final reshape & optional correlation fusion ---
        x_out = x_cross.view(
            batch_size, self.seq_len, self.num_tickers, self.hidden_dim
        )
        x_out = x_out.permute(0, 2, 1, 3).contiguous()
        x_out = x_out.view(batch_size * self.num_tickers, self.seq_len, self.hidden_dim)

        if self.fuse_at_output and corr_in is not None:
            corr_proj = self.corr_out_proj(corr_in)
            x_out = self.output_fusion_block(x_out + corr_proj)

        return x_out


# ==========================================
# ADVANCED TEMPORAL BASELINE (REPLACES LSTM)
# ==========================================


class AdvancedTemporalBaseline(DeepMomentumNetwork):
    """
    A baseline that uses the EXACT same temporal backbone as the main model
    (VSN -> LSTM -> TemporalAttention) but completely lacks the
    Cross-Sectional Attention and GNN layers.

    This effectively models every asset as an independent time-series
    (Independent-In, Independent-Out), serving as a perfect ablation
    for the spatial components.

    This replaces LstmBaseline as the high-capacity temporal benchmark.
    """

    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs,
        )

        self.num_tickers = num_tickers
        self.hidden_dim = hidden_dim
        self.seq_len = kwargs["seq_len"]

        seq_input_dim = self.input_dim + getattr(self, "extra_tcost_channels", 0) + 1

        self.seq_rep = TemporalBackbone(
            num_tickers=num_tickers,
            num_vars=seq_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # EXPLICITLY DISABLE CROSS-SECTIONAL LOGIC
        # This tells DeepMomentumNetwork NOT to attempt to un-flatten
        # the batch dimension or compute portfolio-level Sharpe.
        # It will compute Sharpe on the "pooled" set of [Batch*Assets] independent returns.
        self.is_cross_section = False

    def forward_candidate_arch(
        self,
        target_x,
        target_tickers,
        mask_entire_sequence=None,
        mask_single_date=None,
        batch_size=None,
        corr_features=None,  # Ignored in baseline
        **kwargs,
    ):
        # --- 1. Prepare Inputs ---
        trans_cost_bp = kwargs.get("trans_cost_bp", None)

        # For independent baseline, target_x is [Batch, Seq_Len, Features]
        # or possibly [Batch, Seq_Len] if 1 feature.
        # We rely on DMN passing it as a flat batch of sequences.

        # Ensure it's 3D: [Batch, Seq_Len, Features]
        if target_x.dim() == 2:
            target_x = target_x.unsqueeze(-1)

        actual_batch_size, seq_len, _ = target_x.shape

        # Handle Mask: Default to Zeros (Assume Valid) if missing
        if mask_single_date is None:
            mask_flat = torch.zeros(
                (actual_batch_size, seq_len), device=target_x.device
            )
        else:
            # Mask should match batch dimension
            mask_flat = mask_single_date.view(actual_batch_size, seq_len)

        # Fuse mask feature (Required by input_dim logic)
        x_in = torch.cat([target_x, mask_flat.float().unsqueeze(-1)], dim=-1)

        tickers_flat = target_tickers.view(-1)

        # --- 2. Transaction Cost Context (Matches Main Model) ---
        cost_context = None
        if trans_cost_bp is not None and self.use_transaction_costs:
            # tcost_bp is expected to match input batch dimensions
            # [Batch, Seq, 1]
            tcost_flat = trans_cost_bp.view(actual_batch_size, seq_len, 1)

            if self.logscale_cost_inputs:
                cost_static = (
                    tcost_flat[:, 0, 0] * self.trans_cost_scaler
                ).log1p() / math.log(2)
            else:
                cost_static = tcost_flat[:, 0, 0] * self.trans_cost_scaler

            cost_context = cost_static.reshape(actual_batch_size, 1)

        # --- 3. Temporal Backbone ---
        # Returns [Batch, Seq_Len, Hidden]
        x_rep, _ = self.seq_rep(
            x_in,
            tickers_flat,
            cost_context=cost_context,
        )

        # --- 4. Return ---
        # Return flattened representation [Batch, Seq, Hidden]
        return x_rep

    def variable_importance(
        self, target_x, target_tickers, mask_single_date=None, batch_size=None, **kwargs
    ):
        """Extract variable selection weights for interpretation."""

        # Ensure 3D input
        if target_x.dim() == 2:
            target_x = target_x.unsqueeze(-1)

        actual_batch_size, seq_len, _ = target_x.shape

        # Handle Mask
        if mask_single_date is None:
            mask_flat = torch.zeros(
                (actual_batch_size, seq_len), device=target_x.device
            )
        else:
            mask_flat = mask_single_date.view(actual_batch_size, seq_len)

        x_in = torch.cat([target_x, mask_flat.float().unsqueeze(-1)], dim=-1)

        tickers_flat = target_tickers.view(-1)
        static_context = self.seq_rep.ticker_embedding(tickers_flat)

        _, weights = self.seq_rep.vsn(x_in, static_context)

        # Reshape for consistency [Batch, 1, Seq, Num_Vars] (Fake ticker dim)
        weights = weights.unsqueeze(1)

        return weights
