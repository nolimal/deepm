"""Predictions for Deep Momentum model."""

import os
import re
from typing import List

import pandas as pd

from deepm.backtest.models.traditional import AbstractModel
from deepm.configs.load import load_settings_for_architecture
from deepm.data.dataset import (
    MomentumDataset,
    CrossSectionDataset,
    correlation_features,
    datetime_embedding_global,
)
from deepm.models.base import DmnMode
from deepm.training.train import TrainDeepMomentumNetwork

YEAR_NO_END = 9999

FIELDS_REQUIRED_FOR_PREDICT = [
    "dropout",
    "lr",
    "max_gradient_norm",
    "hidden_dim",
    "num_heads",
    "temporal_att_placement",
    "num_tickers",
    "num_tickers_full_univ",
    "vs_factor_scaler",
    "trans_cost_scaler",
]


class DeepMomentum(AbstractModel):
    """Deep Momentum model."""

    def __init__(
        self,
        train_yaml: str,
        architecture: str,
        top_n_seeds: int,
        seq_len: int,
        pre_loss_steps: int,
        batch_size: int,
        prediction_folder: str = None,
        drop_n_seeds_before_top_n: int = 0,
        use_first_n_valid_seeds: int = None,
    ):
        self.train_settings = load_settings_for_architecture(train_yaml, architecture)
        self.train_settings["seq_len"] = seq_len
        self.train_settings["pre_loss_steps"] = pre_loss_steps
        self.train_settings["batch_size"] = batch_size
        self.train_settings.setdefault("extra_data_pre_steps", 0)

        self.architecture = architecture
        self.top_n_seeds = top_n_seeds
        self.drop_n_seeds_before_top_n = drop_n_seeds_before_top_n
        self.use_first_n_valid_seeds = use_first_n_valid_seeds
        self.seq_len = seq_len
        self.pre_loss_steps = pre_loss_steps
        self.batch_size = batch_size
        self.prediction_folder = prediction_folder

        files = pd.Series(os.listdir(self.directory))
        self.model_start_years = (
            files[files.map(self.is_valid_year)].astype(int).sort_values()
        )

        super().__init__()

    @property
    def directory(self):
        return os.path.join(
            self.train_settings["save_directory"],
            self.train_settings["description"],
            self.train_settings["run_name"],
        )

    def get_model_directory(self, start_year):
        return os.path.join(self.directory, f"{start_year}")

    # ── Prediction ────────────────────────────────────────────────

    def predict(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        variable_importance_mode: bool = False,
        output_signal_weights: bool = False,
    ):
        corr_feat, date_time_embedding = self._build_global_features(data)

        mask = self.model_start_years >= int(start_date[:4])
        start_years = self.model_start_years[mask]
        end_years = start_years.shift(-1).fillna(YEAR_NO_END).astype(int)

        positions_by_version = []
        for start_year, end_year in zip(start_years, end_years):
            all_runs = self._load_seed_runs(start_year)
            tickers_dict = self._read_tickers_dict(start_year, all_runs.index[0])

            if self.drop_n_seeds_before_top_n:
                all_runs = all_runs.iloc[self.drop_n_seeds_before_top_n:]

            test_data = self._build_test_dataset(
                data, tickers, tickers_dict, start_year, end_year,
                corr_feat, date_time_embedding,
            )

            for i in range(self.top_n_seeds):
                positions_by_version.append(
                    self._predict_seed(
                        start_year, all_runs.index[i], test_data
                    ).assign(version=i)
                )

        positions = pd.concat(positions_by_version)
        return self._aggregate_positions(positions, tickers, output_signal_weights)

    def _build_global_features(self, data):
        """Build correlation features and datetime embedding if configured."""
        settings = self.train_settings
        corr_feat = None
        date_time_embedding = None

        if settings.get("use_correlation_features", False):
            corr_feat = correlation_features(
                data,
                settings["correlation_feature"],
                settings["correlation_span"],
                lag=1 if settings.get("use_lagged_cross_section", False) else 0,
            )

        if settings.get("date_time_embedding", False):
            max_len = settings["datetime_embedding_global_max_length"]
            date_time_embedding = datetime_embedding_global(
                settings["first_train_year"], data.index.year.max()
            )
            if len(date_time_embedding) > max_len and not settings["local_time_embedding"]:
                raise ValueError(
                    f"Date time embedding too long: {len(date_time_embedding)} "
                    "- please train with a higher max length or reduce the number of years."
                )

        return corr_feat, date_time_embedding

    def _load_seed_runs(self, start_year):
        """Load and filter the runs CSV for a given start year."""
        all_runs = pd.read_csv(
            os.path.join(self.get_model_directory(start_year), "all_runs.csv"),
            index_col=0,
        )
        all_runs.index = all_runs.index.astype(str)

        if self.use_first_n_valid_seeds:
            all_runs = all_runs.dropna()
            run_number = all_runs.index.map(lambda s: int(s.split("-")[-1]))
            last_valid = run_number.sort_values()[: self.use_first_n_valid_seeds].iloc[-1]
            all_runs = all_runs[run_number <= last_valid]

        return all_runs

    def _read_tickers_dict(self, start_year, run_name):
        """Read the tickers_dict from a run's settings JSON."""
        path = os.path.join(
            self.get_model_directory(start_year), "settings", f"{run_name}.json",
        )
        run_settings = pd.read_json(path, typ="series")
        tickers_dict = pd.Series(run_settings["tickers_dict"])
        tickers_dict.name = "ticker_id"
        tickers_dict.index.name = "ticker"
        return tickers_dict

    def _build_test_dataset(
        self, data, tickers, tickers_dict, start_year, end_year,
        corr_feat, date_time_embedding,
    ):
        """Construct the appropriate dataset for prediction."""
        if self.train_settings["cross_section"]:
            extra_params = {}
            if corr_feat is not None:
                extra_params["corr_features"] = corr_feat
                if self.train_settings.get("use_principal_components", False):
                    extra_params["use_principal_components"] = True

            return CrossSectionDataset(
                self.train_settings,
                data[data.ticker.isin(tickers_dict.index.tolist())],
                first_year=start_year,
                end_year=end_year,
                tickers_dict=tickers_dict,
                test_set=True,
                live_mode=True,
                date_time_embedding=date_time_embedding,
                **extra_params,
            )

        return MomentumDataset(
            self.train_settings,
            data[data.ticker.isin(tickers)],
            first_year=start_year,
            end_year=end_year,
            tickers_dict=tickers_dict,
            test_set=True,
            live_mode=True,
        )

    def _predict_seed(self, start_year, run_name, test_data):
        """Run prediction for a single seed/run."""
        path = os.path.join(
            self.get_model_directory(start_year), "settings", f"{run_name}.json",
        )
        run_settings = pd.read_json(path, typ="series")
        run_settings["num_tickers"] = len(run_settings["tickers_dict"])

        predict_kwargs = {
            k: run_settings[k]
            for k in FIELDS_REQUIRED_FOR_PREDICT
            if k in run_settings
        }

        dmn = TradingSignalDmn(
            self.architecture,
            test_data,
            num_features=len(self.train_settings["features"]),
            model_save_path=self.get_model_directory(start_year),
            parent_class=TrainDeepMomentumNetwork,
            **self.train_settings,
            **predict_kwargs,
        )
        return dmn.predict(run_name, self.batch_size)

    def _aggregate_positions(self, positions, tickers, output_signal_weights):
        """Average positions across seeds and pivot to wide format."""
        if self.train_settings["cross_section"]:
            positions = positions[positions.ticker.isin(tickers)]

        if output_signal_weights:
            return (
                positions.drop(columns="version")
                .reset_index()
                .rename(columns={"index": "date"})
                .groupby(["date", "ticker"])
                .mean()
                .reset_index()
                .set_index("date")
            )

        return (
            positions.groupby([positions.index, positions.ticker])["position"]
            .mean()
            .unstack()
        )

    @classmethod
    def is_valid_year(cls, year: str) -> bool:
        """Check if year string matches YYYY format."""
        return bool(re.match(r"^(19|20)\d{2}$", year))


class TradingSignalDmn:
    """Live trading signal generator for trained Deep Momentum models."""

    def __init__(
        self,
        architecture,
        test_data,
        num_features,
        num_tickers,
        model_save_path,
        parent_class=TrainDeepMomentumNetwork,
        extra_settings=None,
        **kwargs,
    ):
        extra_settings = extra_settings or {}
        self.parent = parent_class(
            None,  # no train data
            None,  # no valid data
            test_data,
            kwargs["seq_len"],
            num_features,
            model_save_path,
            **extra_settings,
        )
        self.optimise_loss_function = kwargs["optimise_loss_function"]
        self.model = self.parent._load_architecture(
            architecture, num_features, num_tickers, **kwargs
        )

    def predict(self, run_name, batch_size):
        return self.parent.predict(
            model=self.model,
            model_save_path=self.parent.model_save_path(run_name),
            batch_size=batch_size,
            optimise_loss_function=self.optimise_loss_function,
            mode=DmnMode.LIVE,
        )