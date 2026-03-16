from deepm.models.base import (
    DeepMomentumNetwork,
    SequenceRepresentation,
    SequenceRepresentationSimple,
)


class LstmBaseline(DeepMomentumNetwork):
    """TFT-style LSTM baseline with VSN and static enrichment."""

    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear: int = 0,
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

        self.seq_rep = SequenceRepresentation(
            self.input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input=False,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
        )

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        representation, _ = self.seq_rep(
            target_x, target_tickers, pos_encoding_batch=pos_encoding_batch,
        )
        return representation

    def variable_importance(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        importance = self.seq_rep(
            target_x, target_tickers,
            pos_encoding_batch=pos_encoding_batch,
            variable_importance=True,
        )
        return importance.squeeze(-2).swapaxes(0, 1)


class LstmSimple(DeepMomentumNetwork):
    """Plain LSTM baseline with optional ticker embedding."""

    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear: int = 0,
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

        self.seq_rep = SequenceRepresentationSimple(
            self.input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input=False,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            use_prescaler=self.date_time_embedding,
        )

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        representation = self.seq_rep(
            target_x, target_tickers, pos_encoding_batch=pos_encoding_batch,
        )
        return representation

    def variable_importance(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        importance = self.seq_rep(
            target_x, target_tickers,
            pos_encoding_batch=pos_encoding_batch,
            variable_importance=True,
        )
        return importance.squeeze(-2).swapaxes(0, 1)
