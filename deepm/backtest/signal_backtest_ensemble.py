"""Job for ensemble signal backtesting."""

import os

import numpy as np
import pandas as pd

from deepm._paths import DATA_DIR, DIAGNOSTICS_DIR
from deepm.backtest.signal_backtest import SignalBacktestJob
from deepm.data.dataset import get_transaction_costs


class EnsembleSignalBacktestJob(SignalBacktestJob):
    """Generate ensemble trading signals and optionally run backtest diagnostics."""

    def __init__(  # pylint: disable=super-init-not-called
        self,
        name: str,
        start_date: str,
        data_parquet: str,
        save_dir: str,
        list_models: dict,
        transaction_cost_path: str,
        variable_importance_mode: bool = False,
        output_signal_weights: bool = False,
        end_date: str = None,
        smooth_signal_span: int = None,
    ):
        """Initialize ensemble signal backtest from multiple model configs."""
        self.name = name
        self.start_date = start_date
        self.data_parquet = data_parquet
        self.save_dir = save_dir
        self.list_models = list_models
        self.transaction_cost_path = transaction_cost_path

        self.load_data_path = str(DATA_DIR / ("feats-" + data_parquet))

        self.end_date = end_date
        self.smooth_signal_span = smooth_signal_span

        self.variable_importance_mode = variable_importance_mode
        if self.variable_importance_mode:
            self.save_dir = os.path.join(self.save_dir, "variable_importance")
        self.output_signal_weights = output_signal_weights
        if self.output_signal_weights:
            self.save_dir = os.path.join(self.save_dir, "signal_weights")

        ticker_sets = []
        for model in list_models:
            cfg_tickers = model["universe"]
            for asset_class in cfg_tickers:
                ticker_sets.append(set(cfg_tickers[asset_class]))
        self.tickers = list(set().union(*ticker_sets))

    def generate_signals(
        self,
        save: bool = True,
    ):
        """Run signal generation for each sub-model and average the ensemble."""
        predictions = []
        for model in self.list_models:
            cfg_tickers = model["universe"]
            ticker_list = list(cfg_tickers.keys())

            config_model = model["model"]
            single_job = SignalBacktestJob(
                self.name,
                ticker_list,
                self.start_date,
                data_parquet=self.data_parquet,
                save_dir=self.save_dir,
                cfg_model=config_model,
                transaction_cost_path=self.transaction_cost_path,
                variable_importance_mode=self.variable_importance_mode,
                output_signal_weights=self.output_signal_weights,
                directory_already_altered=True,
                end_date=self.end_date,
            )
            predictions.append(
                single_job.generate_signals(save=False) * model["weight"]
            )

        predictions = pd.concat(predictions)
        nulls = predictions.groupby(predictions.index).apply(lambda x: x.isnull().all())
        predictions = (
            predictions.groupby(predictions.index).sum().where(~nulls, other=np.nan)
        ).sort_index()

        if self.output_signal_weights:
            predictions.index = pd.MultiIndex.from_tuples(
                predictions.index, names=("date", "ticker")
            )
            if self.smooth_signal_span:
                raise NotImplementedError("Signal smoothing not yet implemented")

        if self.smooth_signal_span:
            predictions = predictions.ewm(span=self.smooth_signal_span).mean()

        if save:
            self.save_predictions(
                predictions,
                self.save_dir,
                self.variable_importance_mode or self.output_signal_weights,
            )
        return predictions

    def main(
        self,
        diagnostics: bool = False,
    ):
        """Run ensemble signal generation and optionally generate diagnostics."""

        dataframe = self.generate_signals(save=True)

        if diagnostics:
            diagnostics_dir = str(DIAGNOSTICS_DIR / self.name)

            reference = pd.read_parquet(
                os.path.join(self.load_data_path, "all.parquet")
            )[["ticker", "target", "vs_factor"]]

            transaction_costs = get_transaction_costs(self.transaction_cost_path)
            positions = self.calc_returns(
                dataframe, transaction_costs, reference, scale_to_target_vol=False
            )
            portfolio_returns = (
                positions.groupby("date")[["gross_return", "net_return"]].sum()
                / positions["ticker"].nunique()
            )
            self.save_and_plot_diagnostics(
                portfolio_returns, positions, diagnostics_dir
            )


