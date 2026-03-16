import math
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from deepm.models.common import (
    VariableSelectionNetwork,
    DropoutNoScaling,
    GateAddNorm,
    GatedResidualNetwork,
    positional_encoding,
    rolling_average_conv,
)


STD_EPSILON = 1e-8


class DmnMode(Enum):
    """Operating mode for Deep Momentum Network inference."""

    TRAINING = 1
    INFERENCE = 2
    LIVE = 3




def sharpe_Lq(
    returns: torch.Tensor,
    q: float = 2.0,
    eps: float = 1e-12,
    annualization_factor: float = 252,
    reduction: str = "mean",
):
    """
    L_q-Norm Sharpe Ratio S_q for [B, S] portfolio returns.
    q=2 is standard Sharpe; q>2 is tail-focused.
    """
    if returns.dim() != 2:
        raise ValueError(f"returns must be [B, S], got {returns.shape}")

    mu = returns.mean(dim=1)
    centered = returns - mu.unsqueeze(1)
    moment_q = (centered.abs() ** q).mean(dim=1)
    sigma_q = (moment_q + eps).pow(1.0 / q)
    S_q = mu / (sigma_q + eps)

    if annualization_factor is not None:
        S_q = S_q * (annualization_factor ** 0.5)

    if reduction == "mean":
        return S_q.mean()
    elif reduction == "none":
        return S_q
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def sharpe_Lq_loss(
    returns: torch.Tensor,
    q: float = 2.0,
    eps: float = 1e-12,
    annualization_factor: float = 252,
):
    """Minimization-friendly loss = -S_q."""
    return -sharpe_Lq(
        returns, q=q, eps=eps,
        annualization_factor=annualization_factor, reduction="mean",
    )


class DeepMomentumNetworkMeta(ABCMeta, type(nn.Module)):
    """Metaclass combining ABCMeta and nn.Module type."""

    pass


class DeepMomentumNetwork(ABC, nn.Module, metaclass=DeepMomentumNetworkMeta):
    """Base class for Deep Momentum Networks (Sharpe-ratio loss)."""

    def __init__(self, input_dim, num_tickers, **kwargs) -> None:
        super().__init__()

        # Core architecture
        self.hidden_dim = kwargs["hidden_dim"]
        self.dropout = kwargs["dropout"]
        self.seq_len = kwargs["seq_len"]
        self.pre_loss_steps = kwargs["pre_loss_steps"]

        self.prediction_head_init = kwargs.get("prediction_head_init", "default")
        self.prediction_head_std = kwargs.get("prediction_head_std", 1e-4)
        self.logscale_cost_inputs = kwargs.get("logscale_cost_inputs", False)
        self.assume_same_leverage_for_prev = kwargs.get("assume_same_leverage_for_prev", False)

        # Feature flags
        self.date_time_embedding = kwargs.get("date_time_embedding", False)
        self.use_transaction_costs = kwargs.get("use_transaction_costs", False)
        self.tcost_inputs = kwargs.get("tcost_inputs", False)
        self.fixed_trans_cost_bp_loss = kwargs.get("fixed_trans_cost_bp_loss", None)
        self.trans_cost_separate_loss = kwargs.get("trans_cost_separate_loss", False)
        self.turnover_regulariser_scaler = kwargs.get("turnover_regulariser_scaler", 0.0)
        self.avg_returns_over = kwargs.get("avg_returns_over", None)
        self.avg_exponentially_weighted = kwargs.get("avg_exponentially_weighted", False)

        # Date-time embedding
        if self.date_time_embedding:
            self.datetime_embedding_global_max_length = kwargs["datetime_embedding_global_max_length"]
            self.local_time_embedding = kwargs.get("local_time_embedding", False)
            self.positional_encoding = positional_encoding(
                self.datetime_embedding_global_max_length, self.hidden_dim
            )

        # Transaction cost scalers
        if self.use_transaction_costs:
            self.vs_factor_scaler = kwargs["vs_factor_scaler"]
            self.trans_cost_scaler = kwargs["trans_cost_scaler"]

        self.extra_tcost_channels = 1 if (self.use_transaction_costs and self.tcost_inputs) else 0

        self.input_dim = input_dim
        self.num_tickers = num_tickers

        # Prediction head: Hidden -> Position [-1, 1]
        no_bias = kwargs.get("no_bias_in_prediction_head", False)
        self.prediction_to_position = nn.Sequential(
            nn.Linear(self.hidden_dim, 1, bias=not no_bias),
            nn.Tanh(),
        )
        self._init_prediction_head(no_bias)

        # Asset dropout
        if "asset_dropout" in kwargs and kwargs["asset_dropout"] > 0.0:
            self.is_asset_dropout = True
            self.asset_dropout = DropoutNoScaling(kwargs["asset_dropout"])
        else:
            self.is_asset_dropout = False

        # Robust / minimax-style Sharpe
        self.use_softmin_sharpe = bool(kwargs.get("use_softmin_sharpe", False))
        self.softmin_sharpe_group_size = kwargs.get("softmin_sharpe_group_size", None)
        self.softmin_sharpe_beta = float(kwargs.get("softmin_sharpe_beta", 10.0))
        self.softmin_sharpe_center = bool(kwargs.get("softmin_sharpe_center", True))
        self.use_joint_pooled_softmin_sharpe = bool(kwargs.get("use_joint_pooled_softmin_sharpe", False))
        self.joint_softmin_lambda = float(kwargs.get("joint_softmin_lambda", 0.1))

        self._last_objective_components = {}

        # Used by backtest pipeline to decide output format
        self.output_signal_weights = kwargs.get("output_signal_weights", False)

        self.is_cross_section = False  # overridden by child classes

    @property
    def device(self):
        return next(self.parameters()).device

    def _init_prediction_head(self, no_bias: bool):
        head_linear = self.prediction_to_position[0]
        mode = self.prediction_head_init

        if mode == "near_zero":
            nn.init.normal_(head_linear.weight, mean=0.0, std=self.prediction_head_std)
            if not no_bias:
                nn.init.constant_(head_linear.bias, 0.0)
        elif mode == "zero":
            nn.init.constant_(head_linear.weight, 0.0)
            if not no_bias:
                nn.init.constant_(head_linear.bias, 0.0)
        elif mode == "default":
            pass
        else:
            raise ValueError(
                f"Unknown prediction_head_init='{mode}'. "
                "Use one of ['near_zero', 'zero', 'default']."
            )

    @abstractmethod
    def forward_candidate_arch(self, target_x, target_tickers, **kwargs):
        pass

    @abstractmethod
    def variable_importance(self, target_x, target_tickers, **kwargs):
        pass

    def concat_transaction_cost_inputs(
        self,
        input_tensor: torch.Tensor,
        vol_scaling: torch.Tensor,
        vol_scaling_prev: torch.Tensor,
        trans_cost_bp: torch.Tensor,
    ):
        """Append scaled vs_factor as an extra dynamic channel."""
        vs = vol_scaling * self.vs_factor_scaler
        if self.logscale_cost_inputs:
            vs = vs.log1p()
        return torch.cat([input_tensor, vs.unsqueeze(-1)], dim=-1)

    def combine_batch_and_asset_dim(self, x, num_dimensions=4):
        if num_dimensions == 2:
            return x.reshape(-1)
        elif num_dimensions == 3:
            return x.reshape(-1, x.shape[-1])
        elif num_dimensions == 4:
            return x.reshape(-1, x.shape[-2], x.shape[-1])
        else:
            raise ValueError(f"Unsupported num_dimensions={num_dimensions}")

    def revert_to_original_dimensions(self, x, num_dimensions_original=4):
        if num_dimensions_original == 3:
            return x.reshape(-1, self.num_tickers, x.shape[-1])
        elif num_dimensions_original == 4:
            return x.reshape(-1, self.num_tickers, x.shape[-2], x.shape[-1])
        else:
            raise ValueError(f"Unsupported num_dimensions_original={num_dimensions_original}")

    @staticmethod
    def softmin_sharpe_loss(
        returns_bt: torch.Tensor,
        beta: float = 10.0,
        group_size: int | None = None,
        annualization_factor: float = 252.0,
        eps: float = 1e-12,
        center: bool = True,
    ) -> torch.Tensor:
        """
        Minimax-style Sharpe: maximize soft-min over per-sample Sharpe.
        Loss = -softmin_beta(Sharpe_b).
        """
        if returns_bt.dim() != 2:
            raise ValueError(f"returns_bt must be [B, T], got {returns_bt.shape}")

        mu = returns_bt.mean(dim=1)
        sigma = returns_bt.std(dim=1, correction=0) + eps
        sharpe_b = (mu / sigma) * math.sqrt(annualization_factor)

        B_total = sharpe_b.shape[0]
        if group_size is None:
            group_size = B_total
        if B_total % group_size != 0:
            raise ValueError(f"Batch {B_total} must be divisible by group_size={group_size}.")

        G = B_total // group_size
        sharpe_gb = sharpe_b.view(G, group_size)

        lse = torch.logsumexp(-beta * sharpe_gb, dim=1)
        if center:
            lse = lse - math.log(group_size)

        softmin = -(lse / beta)
        return -softmin.mean()

    @staticmethod
    def pooled_sharpe_loss(
        returns_bt: torch.Tensor,
        annualization_factor: float = 252.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        mu = returns_bt.mean()
        sigma = returns_bt.std(correction=0) + eps
        sr_pool = (mu / sigma) * math.sqrt(annualization_factor)
        return -sr_pool

    def _apply_asset_dropout(self, kwargs):
        """Apply asset dropout during training — randomly mask out entire assets."""
        if "mask_entire_sequence" not in kwargs:
            kwargs["mask_entire_sequence"] = torch.zeros(
                kwargs["batch_size"], self.num_tickers,
                device=self.device, dtype=torch.bool,
            )

        rand_matrix = torch.rand(kwargs["batch_size"], self.num_tickers, device=self.device)
        dropout_mask = rand_matrix < self.asset_dropout.p
        final_mask = kwargs["mask_entire_sequence"] | dropout_mask

        all_masked = final_mask.all(dim=1)
        if all_masked.any():
            ids_to_keep = rand_matrix[all_masked].argmax(dim=1)
            batch_indices = torch.where(all_masked)[0]
            final_mask[batch_indices, ids_to_keep] = False

        kwargs["mask_entire_sequence"] = final_mask

    def _compute_turnover_cost(
        self, positions, vol_scaling_amount, vol_scaling_amount_prev, trans_cost_bp,
    ):
        """Compute turnover cost for the transaction-cost regulariser."""
        if self.fixed_trans_cost_bp_loss:
            cost_factor = torch.as_tensor(
                self.fixed_trans_cost_bp_loss * 1e-4,
                device=positions.device, dtype=positions.dtype,
            )
        else:
            cost_factor = trans_cost_bp.unsqueeze(-1) * 1e-4

        prev_scale = (
            vol_scaling_amount if self.assume_same_leverage_for_prev
            else vol_scaling_amount_prev
        )

        pos_scaled_cur = positions * vol_scaling_amount.unsqueeze(-1) * cost_factor
        pos_scaled_prev = (
            F.pad(positions, (0, 0, 1, 0))[:, :-1]
            * prev_scale.unsqueeze(-1)
            * cost_factor
        )

        return self.turnover_regulariser_scaler * torch.abs(pos_scaled_cur - pos_scaled_prev)

    def _compute_sharpe_loss(self, returns_slice, q, force_sharpe_loss):
        """Compute the Sharpe-ratio loss (pooled, softmin, joint, or Lq)."""
        if q != 2.0 and (self.use_softmin_sharpe or self.use_joint_pooled_softmin_sharpe):
            raise ValueError("softmin/joint Sharpe currently implemented for q=2.0 only.")

        loss_pool = None
        if q == 2.0:
            loss_pool = self.pooled_sharpe_loss(returns_slice, annualization_factor=252.0, eps=STD_EPSILON)

        loss_softmin = None
        if (self.use_softmin_sharpe or self.use_joint_pooled_softmin_sharpe) and not force_sharpe_loss:
            loss_softmin = self.softmin_sharpe_loss(
                returns_slice,
                beta=self.softmin_sharpe_beta,
                group_size=self.softmin_sharpe_group_size,
                annualization_factor=252.0,
                eps=STD_EPSILON,
                center=self.softmin_sharpe_center,
            )

        if self.use_joint_pooled_softmin_sharpe and not force_sharpe_loss:
            loss = loss_pool + self.joint_softmin_lambda * loss_softmin
        elif self.use_softmin_sharpe and not force_sharpe_loss:
            loss = loss_softmin
        elif q == 2.0:
            loss = loss_pool
        else:
            loss = sharpe_Lq_loss(returns_slice, q=q, annualization_factor=252)

        with torch.no_grad():
            obj_pool = float((-loss_pool).cpu()) if loss_pool is not None else float("nan")
            obj_soft = float((-loss_softmin).cpu()) if loss_softmin is not None else float("nan")
            obj_joint = (
                obj_pool + self.joint_softmin_lambda * obj_soft
                if self.use_joint_pooled_softmin_sharpe else float("nan")
            )
            self._last_objective_components = {
                "SR_pool": obj_pool,
                "SoftMin_SR": obj_soft,
                "Joint_obj": obj_joint,
                "lambda": float(self.joint_softmin_lambda),
                "beta": float(self.softmin_sharpe_beta),
            }

        return loss

    def forward(
        self,
        target_x,
        target_tickers,
        vol_scaling_amount,
        vol_scaling_amount_prev,
        trans_cost_bp,
        mode: DmnMode,
        target_y: Optional[torch.Tensor] = None,
        force_sharpe_loss: bool = False,
        date_time_embedding_index=None,
        q=2.0,
        **kwargs,
    ):
        kwargs = kwargs.copy()
        last_t_steps = self.seq_len - self.pre_loss_steps
        regulariser_loss = 0.0

        # A. Pre-processing
        if self.date_time_embedding:
            if self.local_time_embedding:
                date_time_embedding_index = (
                    date_time_embedding_index - date_time_embedding_index[:, :1]
                )
            kwargs["pos_encoding_batch"] = self.positional_encoding[
                date_time_embedding_index
            ].to(self.device)

        if self.use_transaction_costs and self.tcost_inputs:
            target_x = self.concat_transaction_cost_inputs(
                target_x, vol_scaling_amount, vol_scaling_amount_prev, trans_cost_bp
            )

        if self.is_asset_dropout and self.training:
            self._apply_asset_dropout(kwargs)

        # B. Architecture forward pass → [Batch*Assets, Time, Hidden]
        representation = self.forward_candidate_arch(
            target_x, target_tickers,
            vol_scaling_amount=vol_scaling_amount,
            trans_cost_bp=trans_cost_bp,
            **kwargs,
        )

        if self.is_cross_section:
            representation = self.combine_batch_and_asset_dim(representation, 4)
            if target_y is not None:
                target_y = self.combine_batch_and_asset_dim(target_y, 4)
            vol_scaling_amount = self.combine_batch_and_asset_dim(vol_scaling_amount, 3)
            vol_scaling_amount_prev = self.combine_batch_and_asset_dim(vol_scaling_amount_prev, 3)
            trans_cost_bp = self.combine_batch_and_asset_dim(trans_cost_bp, 3)

        # C. Prediction head → position [-1, 1]
        positions = self.prediction_to_position(representation)

        if self.avg_returns_over:
            positions = rolling_average_conv(
                positions, self.avg_returns_over, self.device,
                exponentially_weighted=self.avg_exponentially_weighted,
            )

        if self.is_cross_section and "mask_entire_sequence" in kwargs:
            mask_flat = kwargs["mask_entire_sequence"].view(-1, 1, 1)
            positions = positions.masked_fill(mask_flat, 0.0)
            if target_y is not None:
                target_y = target_y.masked_fill(mask_flat, 0.0)

        if mode is DmnMode.LIVE:
            return positions[:, -last_t_steps:, -1]

        # D. Captured returns
        captured_returns = positions * target_y

        if mode is DmnMode.INFERENCE:
            return (
                captured_returns[:, -last_t_steps:, -1],
                positions[:, -last_t_steps:, -1],
            )

        # E. Transaction cost regulariser (training only)
        if self.use_transaction_costs:
            turnover_cost = self._compute_turnover_cost(
                positions, vol_scaling_amount, vol_scaling_amount_prev, trans_cost_bp
            )
            if not self.trans_cost_separate_loss:
                captured_returns = captured_returns - turnover_cost
            else:
                regulariser_loss += turnover_cost.mean()

        # F. Portfolio-level aggregation
        if self.is_cross_section:
            captured_returns = self.revert_to_original_dimensions(captured_returns, 4).mean(dim=1)

        returns_slice = captured_returns[:, -last_t_steps:, -1]

        if kwargs.get("return_returns_and_reg", False):
            if not torch.is_tensor(regulariser_loss):
                regulariser_loss = torch.zeros(
                    (), device=returns_slice.device, dtype=returns_slice.dtype
                )
            return returns_slice, regulariser_loss

        # G. Loss
        return self._compute_sharpe_loss(returns_slice, q, force_sharpe_loss) + regulariser_loss


# ---------------------------------------------------------------------------
# Legacy sequence representations (used by LSTM / MomentumTransformer baselines)
# ---------------------------------------------------------------------------


class SequenceRepresentationSimple(nn.Module):
    """Simple LSTM-based sequence encoder with optional ticker embedding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_tickers: int,
        fuse_encoder_input: bool = False,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,
        use_prescaler=False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tickers = num_tickers
        self.use_static_ticker = use_static_ticker
        self.use_prescaler = use_prescaler

        if auto_feature_num_input_linear:
            raise NotImplementedError("auto_feature_num_input_linear not supported")
        if fuse_encoder_input:
            raise NotImplementedError("fuse_encoder_input not supported")

        if use_static_ticker:
            self.ticker_embedding = nn.Embedding(self.num_tickers, self.hidden_dim)
            self.static_context_variable_selection = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=dropout
            )
            self.static_context_state_h = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=dropout
            )
            self.static_context_state_c = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=dropout
            )

        if self.use_prescaler:
            self.prescaler = nn.Linear(input_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(self.input_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        static_tickers: torch.Tensor,
        encoder_representation: Union[torch.Tensor, None] = None,
        automatic_features: Union[torch.Tensor, None] = None,
        variable_importance: bool = False,
        pos_encoding_batch: Union[torch.Tensor, None] = None,
    ):
        sequence = sequence.swapaxes(0, 1)
        if self.use_prescaler:
            sequence = self.prescaler(sequence)

        if variable_importance:
            raise NotImplementedError("variable_importance not supported for Simple variant")

        if pos_encoding_batch is not None:
            sequence = sequence + pos_encoding_batch.swapaxes(0, 1)

        if self.use_static_ticker:
            ticker_embedding = self.ticker_embedding(static_tickers)
            hidden_state, _ = self.lstm(
                sequence,
                (
                    self.static_context_state_h(ticker_embedding).unsqueeze(0),
                    self.static_context_state_c(ticker_embedding).unsqueeze(0),
                ),
            )
        else:
            hidden_state, _ = self.lstm(sequence)

        hidden_state = self.dropout(hidden_state)
        return hidden_state.swapaxes(0, 1)


class SequenceRepresentation(nn.Module):
    """TFT-style sequence encoder with VSN, LSTM, and static enrichment."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_tickers: int,
        fuse_encoder_input: bool = False,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_tickers = num_tickers
        self.encoder_input = fuse_encoder_input
        self.use_static_ticker = use_static_ticker

        if fuse_encoder_input:
            self.combine_layer = GatedResidualNetwork(
                2 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout
            )

        if auto_feature_num_input_linear:
            prescalers = nn.ModuleDict(
                dict(
                    [(str(i), nn.Linear(1, hidden_dim)) for i in range(auto_feature_num_input_linear)]
                    + [(str(i), nn.Identity()) for i in range(auto_feature_num_input_linear, input_dim)]
                )
            )
            self.vsn = VariableSelectionNetwork(
                dict([(str(i), self.hidden_dim) for i in range(input_dim)]),
                self.hidden_dim,
                dropout=self.dropout,
                context_size=self.hidden_dim,
                prescalers=prescalers,
            )
        else:
            self.vsn = VariableSelectionNetwork(
                dict([(str(i), self.hidden_dim) for i in range(input_dim)]),
                self.hidden_dim,
                dropout=self.dropout,
                context_size=self.hidden_dim,
                prescalers={},
            )

        if use_static_ticker:
            self.ticker_embedding = nn.Embedding(self.num_tickers, self.hidden_dim)
            self.static_context_variable_selection = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_enrichment = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_state_h = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_state_c = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.gate_add_norm_lstm = GateAddNorm(hidden_dim, dropout=dropout)
        self.grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim,
            dropout=dropout,
            context_size=hidden_dim if self.use_static_ticker else None,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        static_tickers: torch.Tensor,
        encoder_representation: Union[torch.Tensor, None] = None,
        automatic_features: Union[torch.Tensor, None] = None,
        variable_importance: bool = False,
        pos_encoding_batch: Union[torch.Tensor, None] = None,
    ):
        if self.use_static_ticker:
            ticker_embedding = self.ticker_embedding(static_tickers)

        if automatic_features:
            raise NotImplementedError("automatic_features not supported")

        inputs = dict(
            [
                (str(i), sequence[:, :, i : (i + 1)].swapaxes(0, 1))
                for i in range(self.input_dim)
            ]
        )

        if variable_importance:
            _, importance = self.vsn(
                inputs,
                self.static_context_variable_selection(ticker_embedding)
                if self.use_static_ticker else None,
            )
            return importance

        vsn_out, _ = self.vsn(
            inputs,
            self.static_context_variable_selection(ticker_embedding)
            if self.use_static_ticker else None,
        )

        if pos_encoding_batch is not None:
            vsn_out = vsn_out + pos_encoding_batch.swapaxes(0, 1)

        if self.encoder_input:
            vsn_out = self.combine_layer(
                torch.cat([vsn_out, encoder_representation.swapaxes(0, 1)], dim=-1)
            )

        if self.use_static_ticker:
            hidden_state, _ = self.lstm(
                vsn_out,
                (
                    self.static_context_state_h(ticker_embedding).unsqueeze(0),
                    self.static_context_state_c(ticker_embedding).unsqueeze(0),
                ),
            )
        else:
            hidden_state, _ = self.lstm(vsn_out)

        hidden_state = self.gate_add_norm_lstm(hidden_state, vsn_out)

        ffn_output = self.grn(
            hidden_state,
            self.static_context_enrichment(ticker_embedding)
            if self.use_static_ticker else None,
        )

        ffn_output = ffn_output.swapaxes(0, 1)
        hidden_state = hidden_state.swapaxes(0, 1)

        return ffn_output, hidden_state
