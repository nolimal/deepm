from torch import nn

from deepm.models.common import (
    causal_attention_mask,
    GateAddNorm,
    GatedResidualNetwork,
)
from deepm.models.base import (
    DeepMomentumNetwork,
    SequenceRepresentation,
)


class MomentumTransformer(DeepMomentumNetwork):
    """Momentum Transformer with self-attention and GRN post-processing."""

    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )

        assert isinstance(num_heads, int)
        self.num_heads = num_heads

        self.seq_rep = SequenceRepresentation(
            self.input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input=False,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
        )

        self.gate_add_norm_mha = GateAddNorm(hidden_dim, dropout=dropout)
        self.ffn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )
        self.self_att = nn.MultiheadAttention(
            hidden_dim, self.num_heads, batch_first=True,
        )
        self.gate_add_norm_block = GateAddNorm(hidden_dim, dropout=dropout)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        representation, lstm_hidden_state = self.seq_rep(
            target_x, target_tickers, pos_encoding_batch=pos_encoding_batch
        )

        mask = causal_attention_mask(self.seq_len).to(representation.device)
        mha, _ = self.self_att(
            representation, representation, representation, attn_mask=mask
        )
        add = self.gate_add_norm_mha(mha, representation)
        ffn_representation = self.ffn(add)
        transformer_representation = self.gate_add_norm_block(
            ffn_representation, lstm_hidden_state
        )
        return transformer_representation

    def variable_importance(self, target_x, target_tickers, **kwargs):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)
