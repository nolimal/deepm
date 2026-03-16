import logging
from typing import Union

import torch
import torch.utils.data
import pandas as pd
import numpy as np

from deepm._paths import TCOST_DIR

logger = logging.getLogger(__name__)


def get_transaction_costs(file_path: str) -> pd.DataFrame:
    """
    Load ticker metadata / transaction-cost table.

    Expected to contain at least:
      - 'ticker'
      - 'transaction_cost'

    Can also contain:
      - 'close_utc', 'open_utc', 'close_group', etc.
    """
    costs = pd.read_csv(
        TCOST_DIR / (file_path + ".csv"),
        index_col=0,
    ).reset_index()

    if "ticker" not in costs.columns:
        first_col = costs.columns[0]
        costs = costs.rename(columns={first_col: "ticker"})

    return costs


def calc_vs_factor_scaler(data_ex_test: pd.DataFrame) -> float:
    """Compute inverse-std scaler for the volatility-scaling factor."""
    vs_factor_scaler = (
        1.0 / data_ex_test["vs_factor"].dropna().values.astype(np.float32).std()
    )
    return vs_factor_scaler


def calc_trans_cost_scaler(ticker_ref: pd.DataFrame) -> float:
    """Compute inverse-max scaler for transaction costs."""
    trans_cost_scaler = 1.0 / ticker_ref["transaction_cost"].max()
    return trans_cost_scaler


def datetime_embedding_global(start_year: int, end_year: int) -> pd.Series:
    """Create a global datetime embedding for the given years."""

    date_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year+1}-01-01")
    date_range = date_range[date_range.weekday < 5]  # pylint: disable=no-member
    return pd.Series(list(range(len(date_range))), index=date_range)


def correlation_features(
    data: pd.DataFrame,
    feature: str = "r1d",
    span: int = 252,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Calculates pairwise rolling correlations with an optional lag.

    Args:
        data: DataFrame with index (Date) or Column (Date) and 'ticker' column.
        feature: The column name to calculate correlation on (e.g., returns).
        span: Span for the EWM calculation.
        lag: Number of steps to shift the correlation matrix forward.
             lag=1 means the correlation at time t is calculated using data up to t-1.
    """

    # 1. Pivot to Wide Format (Date x Ticker)
    # Assumes input index is Date. If 'date' is a column, ensure it's handled.
    if "date" in data.columns:
        data = data.set_index("date")

    df_wide = (
        data.pivot(columns="ticker", values=feature).sort_index().ffill().fillna(0.0)
    )

    # 2. Calculate Rolling/EWM Correlation
    # Result is a MultiIndex DataFrame: (Date, Ticker) x (Ticker)
    # Shape: [Num_Dates * Num_Tickers, Num_Tickers]
    corrs = df_wide.ewm(span=span, min_periods=np.minimum(10, span)).corr()

    # 3. Apply Lag (Safety Shift)
    if lag > 0:
        # We unstack to move 'ticker' to columns, leaving only 'date' in the index.
        # This makes the DataFrame 2D: [Date, Ticker_i * Ticker_j]
        # We then shift the index (Time) and stack it back.
        idx = corrs.index
        corrs = (
            corrs.unstack(level=1)  # Move 'ticker' level to columns
            .shift(lag)  # Shift the 'date' index
            .stack()  # Restore original MultiIndex structure
            .reindex(idx)  # Reindex to original structure to fill NaNs
        )


    return corrs


class MomentumDataset(torch.utils.data.Dataset):
    """Per-ticker sequential dataset for single-asset models."""

    def __init__(
        self,
        settings,
        data,
        first_year=2015,
        end_year=2020,
        keep_first_perc=None,
        drop_first_perc=None,
        tickers_dict=None,
        test_set=False,
        live_mode=False,
        date_time_embedding=None,
        target_override=None,
        **kwargs,
    ):
        super().__init__()
        data = data.copy()
        try:
            if keep_first_perc:
                assert not drop_first_perc
                assert not test_set
        except ValueError:
            logger.error("Options for train/valid/test are mutually exclusive")
            raise

        if "stride" in settings and settings["stride"] != 1:
            raise NotImplementedError("Stride != 1 is not yet supported")

        self.settings = settings
        self.seq_len = settings["seq_len"]

        self.num_features = len(settings["features"])
        self.num_targets = 1
        self.pre_loss_steps = (
            settings["pre_loss_steps"] + settings["extra_data_pre_steps"]
        )

        pred_steps = self.seq_len - self.pre_loss_steps
        self.pred_steps = pred_steps

        if target_override is not None:
            data["target"] = data[target_override]

        if date_time_embedding is not None:
            data = data.loc[data.index.isin(date_time_embedding.index)]

        if live_mode:
            if not test_set:
                test_set = True
                logger.warning("Defaulting to test_set=True in live mode")
            data["target"] = 0.0

        data["date"] = data.index

        if "fillna_features" in settings:
            if "ffill" in settings["fillna_features"]:
                fill_cols = settings["fillna_features"]["ffill"]
                data[fill_cols] = data[fill_cols].ffill()

            if "value" in settings["fillna_features"]:
                data = data.fillna(settings["fillna_features"]["value"])

        if "dataset_filter_year" in settings:
            data = data[data.index.year >= settings["dataset_filter_year"]]

        ticker_ref = get_transaction_costs(settings["ticker_reference_file"])
        if settings.get("ticker_mapping", None) is not None:
            ticker_ref["ticker"] = ticker_ref["ticker"].map(settings["ticker_mapping"])

        data = data[data.ticker.isin(ticker_ref["ticker"])]

        extra_cols = ["vs_factor", "vs_factor_prev", "transaction_cost", "ticker_id"]
        data["vs_factor_prev"] = data.groupby("ticker")["vs_factor"].shift(1).copy()

        data["date"] = pd.to_datetime(data["date"])
        data = data.reset_index(drop=True).sort_values(["ticker", "date"])

        buffer = self.seq_len if test_set else self.pre_loss_steps
        if first_year:
            data = data[
                data.groupby("ticker")["date"].shift(-buffer + 1).ffill().dt.year
                >= first_year
            ]

        if end_year:
            data = data[data["date"].dt.year < end_year]

        if first_year or end_year:
            data = data.reset_index(drop=True)

        self.num_tickers_full_univ = data["ticker"].nunique()

        data = (
            data[
                settings["reference_cols"]
                + settings["features"]
                + list(set(extra_cols) - {"ticker_id", "transaction_cost"})
                + [settings["target"]]
            ]
            .dropna()
            .copy()
        )

        self.ticker_ref = ticker_ref

        if tickers_dict is None:
            data = data.merge(
                pd.DataFrame({"ticker": data["ticker"].unique()})
                .reset_index()
                .rename(columns={"index": "ticker_id"})
            )
            self.tickers_dict = (
                data[["ticker", "ticker_id"]]
                .drop_duplicates()
                .set_index("ticker")["ticker_id"]
            )
        else:
            self.tickers_dict = tickers_dict
            data = data.merge(tickers_dict.to_frame(), on="ticker")

        data = data.merge(ticker_ref[["ticker", "transaction_cost"]], on="ticker")

        data = (
            data[
                settings["reference_cols"]
                + settings["features"]
                + extra_cols
                + [settings["target"]]
            ]
            .reset_index(drop=True).copy()
        )

        self.tickers = data["ticker_id"].values
        self.num_tickers = data["ticker_id"].nunique()

        self.inputs = torch.Tensor(data[settings["features"]].values.copy())

        self.outputs = torch.Tensor(data[settings["targets"]].values.copy())
        self.date_mask = torch.Tensor(
            (
                (data["date"].dt.year >= first_year) & (data["date"].dt.year < end_year)
            ).values.copy()
        ).bool()

        self.dates = data["date"].dt.strftime("%Y-%m-%d").values

        self.vs_factor = data["vs_factor"].values.astype(np.float32)
        self.vs_factor_prev = data["vs_factor_prev"].values.astype(np.float32)


        self.trans_cost_bp = data["transaction_cost"].values.astype(np.float32)

        self.seq_indexes = self._build_seq_indexes(
            data, keep_first_perc, drop_first_perc
        )

        if date_time_embedding is not None:
            self.date_time_embedding_index = date_time_embedding.loc[
                self.dates.tolist()
            ].values
        else:
            self.date_time_embedding_index = None

    def _build_seq_indexes(self, data, keep_first_perc, drop_first_perc):
        """Partition each ticker's time series into non-overlapping sequences."""
        pred_steps = self.pred_steps
        data["cum_count"] = data.groupby("ticker").cumcount()
        data["count"] = data.groupby(["ticker"])["cum_count"].transform("count")

        prediction_day = data[data["cum_count"] >= self.pre_loss_steps][
            ["date", "ticker", "cum_count", "count"]
        ].copy()

        prediction_day["pred_cum_count"] = prediction_day["cum_count"] - self.pre_loss_steps
        prediction_day["pred_count"] = prediction_day["count"] - self.pre_loss_steps
        prediction_day["num_seqs"] = prediction_day["pred_count"] // pred_steps

        offset = (prediction_day.groupby("ticker").transform("max")["pred_count"]) % (
            prediction_day["num_seqs"] * pred_steps
        )

        prediction_day["pred_cum_count"] = prediction_day["cum_count"] - self.pre_loss_steps + 1
        prediction_day["seq_num"] = (prediction_day["pred_cum_count"] - offset - 1) // pred_steps
        prediction_day = prediction_day[prediction_day["seq_num"] >= 0].copy()

        assert not (len(prediction_day) % pred_steps)

        prediction_day["portion"] = prediction_day["seq_num"] / (prediction_day["num_seqs"] - 1)

        if keep_first_perc:
            prediction_day = prediction_day[prediction_day["portion"] < keep_first_perc]
        if drop_first_perc:
            prediction_day = prediction_day[prediction_day["portion"] >= drop_first_perc]

        seq_indexes = (
            prediction_day.reset_index()
            .groupby(["ticker", "seq_num"])["index"]
            .last()
            .map(lambda i: list(range(i - self.seq_len + 1, i + 1)))
        )
        return seq_indexes.tolist()

    def __getitem__(self, index):
        return (
            self.inputs[self.seq_indexes[index]],
            self.outputs[self.seq_indexes[index]],
            self.date_mask[self.seq_indexes[index]],
            self.dates[self.seq_indexes[index]].tolist(),
            self.tickers[self.seq_indexes[index][0]],
            self.vs_factor[self.seq_indexes[index]],
            self.vs_factor_prev[self.seq_indexes[index]],
            self.trans_cost_bp[self.seq_indexes[index]],
            (
                []
                if self.date_time_embedding_index is None
                else self.date_time_embedding_index[self.seq_indexes[index]]
            ),
            [],
        )

    def __len__(self):
        return len(self.seq_indexes)

    @property
    def tickers_from_idx_dict(self):
        return {value: key for key, value in self.tickers_dict.items()}


class CrossSectionDataset(torch.utils.data.Dataset):
    """Cross-sectional sequence dataset for DeepM/DMN-style models."""

    def __init__(
        self,
        settings,
        data,
        first_year=2015,
        end_year=2020,
        keep_first_perc=None,
        drop_first_perc=None,
        tickers_dict=None,
        test_set=False,
        live_mode=False,
        corr_features=None,
        use_principal_components=False,
        date_time_embedding=None,
        target_override=None,
    ):
        super().__init__()

        stride = settings.get("stride", None)
        data = data.copy()

        if target_override is not None:
            data["target"] = data[target_override]

        # Train/valid/test split sanity checks
        try:
            if keep_first_perc:
                assert not drop_first_perc
                assert not test_set
        except ValueError:
            logger.error("Options for train/valid/test are mutually exclusive")
            raise

        self.settings = settings
        self.seq_len = settings["seq_len"]
        self.num_features = len(settings["features"])
        self.num_targets = 1

        self.pre_loss_steps = (
            settings["pre_loss_steps"] + settings["extra_data_pre_steps"]
        )
        pred_steps = self.seq_len - self.pre_loss_steps
        self.pred_steps = pred_steps

        # Step size for sliding window
        self.step_size = stride if stride is not None else self.pred_steps

        # ---------------------------------------------------------------------
        # 1) Basic setup
        # ---------------------------------------------------------------------
        data.index = pd.to_datetime(data.index)
        data["date"] = data.index

        if live_mode:
            if not test_set:
                test_set = True
                logger.warning("Defaulting to test_set=True in live mode")
            # In live mode we don't have realised target
            data["target"] = 0.0

        ticker_ref, canonical_tickers, data = self._build_ticker_universe(settings, data)

        # extra columns we want to carry
        extra_cols = ["vs_factor", "vs_factor_prev", "transaction_cost", "ticker_id"]
        data["vs_factor_prev"] = data.groupby("ticker")["vs_factor"].shift(1).copy()

        data["date"] = pd.to_datetime(data["date"])
        data = data.reset_index(drop=True).sort_values(["ticker", "date"])

        # ---------------------------------------------------------------------
        # 3) Time filtering (first_year / end_year) in terms of decision dates
        # ---------------------------------------------------------------------
        buffer = self.seq_len if test_set else self.pre_loss_steps

        if first_year:
            # keep rows whose last usable step in the window falls in/after first_year
            data = data[
                data.groupby("ticker")["date"].shift(-buffer + 1).ffill().dt.year
                >= first_year
            ]

        if end_year:
            data = data[data["date"].dt.year < end_year]

        if first_year or end_year:
            data = data.reset_index(drop=True)

        # ---------------------------------------------------------------------
        # 4) Select core columns and basic NA handling
        # ---------------------------------------------------------------------
        self.ticker_ref = ticker_ref

        data = data[
            settings["reference_cols"]
            + settings["features"]
            + list(set(extra_cols) - {"ticker_id", "transaction_cost"})
            + [settings["target"]]
        ]

        # drop rows that still contain NaNs (e.g. at very start of each series)
        data = data.dropna().copy()

        # ---------------------------------------------------------------------
        # 5) Canonical ticker_id mapping (shared across splits and matching adjacency)
        # ---------------------------------------------------------------------
        if tickers_dict is None:
            # First dataset (e.g. training): build mapping from canonical_tickers
            self.tickers_dict = pd.Series(
                data=np.arange(len(canonical_tickers), dtype=int),
                index=pd.Index(canonical_tickers, name="ticker"),
                name="ticker_id",
            )
        else:
            # Subsequent datasets (valid/test): reuse mapping so ticker_ids are consistent
            self.tickers_dict = tickers_dict

        # Merge ticker_id into data using canonical mapping
        data = data.merge(self.tickers_dict.to_frame().reset_index(), on="ticker")

        # Merge transaction costs
        data = data.merge(
            ticker_ref[["ticker", "transaction_cost"]].drop_duplicates(), on="ticker"
        )

        # Re-pack with all columns we care about
        data = (
            data[
                settings["reference_cols"]
                + settings["features"]
                + extra_cols
                + [settings["target"]]
            ]
            .reset_index(drop=True)
            .copy()
        )

        # "Full universe" size is defined by canonical mapping (important for adjacency)
        self.num_tickers_full_univ = len(self.tickers_dict)
        self.num_tickers = self.num_tickers_full_univ

        # ---------------------------------------------------------------------
        # 6) Build complete (ticker, date) grid so GNN sees fixed N across time
        #     Now 'date' means *decision date* after any close-time shift.
        # ---------------------------------------------------------------------
        tickers_index = self.tickers_dict.index  # canonical order (alphabetical)
        dates_index = (
            data["date"].drop_duplicates().sort_values().reset_index(drop=True)
        )

        # Cartesian product of canonical universe and all dates in this period
        multi_index = pd.MultiIndex.from_product(
            [tickers_index, dates_index], names=["ticker", "date"]
        )

        # Left join original data onto the full grid
        data_filled = (
            pd.DataFrame(index=multi_index)
            .join(data.set_index(["ticker", "date"]))
            .reset_index()
            .set_index(["ticker", "date"])
        )

        # Targets: missing -> 0 (we'll mask them explicitly)
        data_filled[settings["targets"]] = data_filled[settings["targets"]].fillna(0.0)

        # ---------------------------------------------------------------------
        # 7) Feature-missing masks + forward fill within ticker
        # ---------------------------------------------------------------------
        # raw missing mask BEFORE any ffill
        self.features_missing = (
            data_filled[settings["features"]].isna().any(axis=1).values
        )

        # forward-fill along each ticker
        data_filled = data_filled.groupby("ticker").ffill()

        # any rows that are *still* missing after ffill are "never observed"
        self.initial_features_missing = (
            data_filled[settings["features"]].isna().any(axis=1).values
        )

        # for those, explicitly set features and targets to NaN so we can zero them and mask out
        data_filled.loc[
            self.initial_features_missing, settings["features"] + settings["targets"]
        ] = np.nan

        # final fill with zeros for the actual tensor values
        data_filled = data_filled.fillna(0.0)

        # Now attach ticker_id again (ensuring canonical ordering)
        data_filled = (
            data_filled.reset_index()
            .drop(columns="ticker_id", errors="ignore")
            .merge(self.tickers_dict.to_frame().reset_index(), on="ticker")
            .set_index("date")
        )

        # mapping from date -> integer index in the flattened panel
        date_mapping = pd.Series(range(len(data_filled)), index=data_filled.index)

        self.seq_indexes = self._build_sliding_windows(
            dates_index, date_mapping, keep_first_perc, drop_first_perc
        )

        self._build_correlation_features(
            corr_features, use_principal_components, multi_index
        )


        # ---------------------------------------------------------------------
        # 10) Final tensors
        # ---------------------------------------------------------------------
        self.inputs = torch.Tensor(np.asarray(data_filled[settings["features"]].values, dtype=np.float32))
        self.outputs = torch.Tensor(np.asarray(data_filled[settings["targets"]].values, dtype=np.float32))

        # optionally reweight outputs for a subset of tickers
        if self.settings.get("output_ticker_subset", None):
            output_weight = (
                (
                    data_filled["ticker_id"]
                    .isin(
                        self.tickers_dict.loc[
                            self.settings["output_ticker_subset"]
                        ].tolist()
                    )
                    .astype(int)
                )
                * self.num_tickers
                / len(self.settings["output_ticker_subset"])
            )
            output_weight = torch.Tensor(output_weight.values).unsqueeze(-1)
            self.outputs *= output_weight

        # date mask only depends on calendar dates (applied to all tickers)
        self.date_mask = torch.Tensor(
            (
                (data_filled.index.year >= first_year)
                & (data_filled.index.year < end_year)
            )
        ).bool()

        # keep date metadata
        self.dates_string = np.asarray(data_filled.index.strftime("%Y-%m-%d"))
        self.dates = pd.Index(dates_index).get_indexer(data_filled.index)

        # ticker ids in canonical order
        self.tickers = data_filled["ticker_id"].values

        self.vs_factor = data_filled["vs_factor"].values.astype(np.float32)
        self.vs_factor_prev = data_filled["vs_factor_prev"].values.astype(np.float32)

        self.trans_cost_bp = data_filled["transaction_cost"].values.astype(np.float32)

        if (
            "date_time_embedding_index" in self.settings
            and self.settings["date_time_embedding_index"]
        ):
            raise NotImplementedError("Date-time embedding index is not yet supported for CrossSectionDataset")

    @staticmethod
    def _hhmm_to_minutes(s: str) -> int:
        h, m = map(int, s.split(":"))
        return 60 * h + m

    def _build_ticker_universe(self, settings, data):
        """Load ticker metadata, compute close-time rankings, apply date shifts."""
        ticker_ref = get_transaction_costs(settings["ticker_reference_file"])
        if settings.get("ticker_mapping", None) is not None:
            ticker_ref["ticker"] = ticker_ref["ticker"].map(settings["ticker_mapping"])
        ticker_ref = ticker_ref.dropna(subset=["ticker"])

        use_close_time_context = settings.get("use_close_time_context", False)
        decision_time_utc = settings.get("decision_time_utc", "21:00")

        if use_close_time_context:
            if "close_utc" not in ticker_ref.columns:
                raise ValueError(
                    "use_close_time_context=True but 'close_utc' not found in ticker_ref. "
                    "Ensure ticker_reference_file has open_utc/close_utc columns."
                )
            decision_minutes = self._hhmm_to_minutes(decision_time_utc)
            close_minutes = ticker_ref["close_utc"].apply(self._hhmm_to_minutes)
            ticker_ref["context_shift_days"] = (close_minutes > decision_minutes).astype(int)
        else:
            ticker_ref["context_shift_days"] = 0

        canonical_tickers = sorted(ticker_ref["ticker"].unique().tolist())

        if "close_utc" in ticker_ref.columns:
            ticker_meta = (
                ticker_ref.drop_duplicates(subset=["ticker"])
                .set_index("ticker")
                .reindex(canonical_tickers)
            )
            if ticker_meta["close_utc"].isna().any():
                missing = ticker_meta[ticker_meta["close_utc"].isna()].index.tolist()
                raise ValueError(f"Missing close_utc for tickers: {missing}")

            close_minutes_vec = ticker_meta["close_utc"].apply(self._hhmm_to_minutes).values
            unique_times, inv = np.unique(close_minutes_vec, return_inverse=True)
            self.ticker_close_rank = inv.astype(np.int64)
            self.num_close_groups = int(len(unique_times))
        else:
            self.ticker_close_rank = np.arange(len(canonical_tickers), dtype=np.int64)
            self.num_close_groups = len(canonical_tickers)

        data = data[data.ticker.isin(canonical_tickers)]
        data = data.merge(
            ticker_ref[["ticker", "context_shift_days"]], on="ticker", how="left"
        )
        data["context_shift_days"] = data["context_shift_days"].fillna(0).astype(int)
        data["date"] = pd.to_datetime(data["date"]) + pd.to_timedelta(
            data["context_shift_days"], unit="D"
        )
        data.drop(columns=["context_shift_days"], inplace=True)

        return ticker_ref, canonical_tickers, data

    def _build_sliding_windows(self, dates_index, date_mapping, keep_first_perc, drop_first_perc):
        """Generate sliding-window sequence indices over the date grid."""
        total_dates = len(dates_index)
        valid_start_indices = list(range(0, total_dates - self.seq_len + 1, self.step_size))
        num_seqs = len(valid_start_indices)

        if keep_first_perc:
            valid_start_indices = valid_start_indices[:int(num_seqs * keep_first_perc)]
        elif drop_first_perc:
            valid_start_indices = valid_start_indices[int(num_seqs * drop_first_perc):]

        seq_indexes = []
        for start_idx in valid_start_indices:
            subset_dates = dates_index.iloc[start_idx : start_idx + self.seq_len]
            seq_indexes.append(date_mapping.loc[subset_dates].tolist())

        seq_indexes = np.array(seq_indexes)

        # Reorder so reshape yields [num_tickers, seq_len, ...]
        if len(seq_indexes) > 0:
            new_order = sorted(
                list(range(seq_indexes.shape[1])),
                key=lambda x: x % self.num_tickers,
            )
            return seq_indexes[:, new_order].tolist()
        return []

    def _build_correlation_features(self, corr_features, use_principal_components, multi_index):
        """Align rolling correlation features to the (ticker, date) grid."""
        if corr_features is not None:
            self.has_corr_features = True
            corr_features = corr_features.loc[
                corr_features.index.get_level_values(1).isin(self.tickers_dict.index),
                (corr_features.columns if use_principal_components else self.tickers_dict.index),
            ]
            self.corr_features = torch.Tensor(
                corr_features.swaplevel(0, 1)
                .reindex(multi_index)
                .fillna(0.0)
                .values
            )
            self.corr_features_dim = self.corr_features.shape[1]
        else:
            self.has_corr_features = False
            self.corr_features_dim = 0

    def __getitem__(self, index):
        return (
            self.inputs[self.seq_indexes[index]].reshape(
                self.num_tickers, self.seq_len, self.num_features
            ),
            self.outputs[self.seq_indexes[index]].reshape(
                self.num_tickers, self.seq_len, self.num_targets
            ),
            self.date_mask[self.seq_indexes[index]].reshape(
                self.num_tickers, self.seq_len
            ),
            self.dates[self.seq_indexes[index]].reshape(self.num_tickers, self.seq_len),
            self.tickers[self.seq_indexes[index]][
                range(0, self.seq_len * self.num_tickers, self.seq_len)
            ],
            self.vs_factor[self.seq_indexes[index]].reshape(
                self.num_tickers, self.seq_len
            ),
            self.vs_factor_prev[self.seq_indexes[index]].reshape(
                self.num_tickers, self.seq_len
            ),
            self.trans_cost_bp[self.seq_indexes[index]].reshape(
                self.num_tickers, self.seq_len
            ),
            ([]),
            ([]),
            self.initial_features_missing[self.seq_indexes[index]].reshape(
                self.num_tickers, self.seq_len
            )[:, 0],
            self.features_missing[self.seq_indexes[index]].reshape(
                self.num_tickers, self.seq_len
            ),
            (
                torch.Tensor([])
                if not self.has_corr_features
                else self.corr_features[self.seq_indexes[index]].reshape(
                    self.num_tickers, self.seq_len, self.corr_features_dim
                )
            ),
        )

    def __len__(self):
        return len(self.seq_indexes)

    @property
    def tickers_from_idx_dict(self):
        return {value: key for key, value in self.tickers_dict.items()}



def unpack_torch_dataset(
    samples,
    dataset: Union[MomentumDataset, CrossSectionDataset],
    device: torch.device,
    use_dates_mask: bool = False,
    live_mode: bool = False,
):
    """Unpack a batch of samples from a MomentumDataset or CrossSectionDataset."""
    (
        target_x,
        target_y,
        date_mask,
        dates,
        target_tickers,
        vol_scaling_amount,
        vol_scaling_amount_prev,
        trans_cost_bp,
        date_time_embedding_index,
        to_weight_x,
    ) = samples[:10]

    target_x = target_x.to(device)
    target_tickers = target_tickers.to(device)

    vol_scaling_amount = vol_scaling_amount.to(device)
    vol_scaling_amount_prev = vol_scaling_amount_prev.to(device)
    trans_cost_bp = trans_cost_bp.to(device)

    if live_mode:
        target_y = None
    else:
        target_y = target_y.to(device)

    if use_dates_mask:
        date_mask = date_mask.to(device)
    else:
        date_mask = None

    if date_time_embedding_index == []:
        date_time_embedding_index = None

    if to_weight_x == []:
        to_weight_x = None
    else:
        to_weight_x = to_weight_x.to(device)

    extra_data = {}

    if isinstance(dataset, CrossSectionDataset):
        (
            mask_entire_sequence,
            mask_single_date,
            corr_features,
        ) = samples[10:13]

        mask_entire_sequence = mask_entire_sequence.to(device)
        mask_single_date = mask_single_date.to(device)
        if corr_features is not None:
            corr_features = corr_features.to(device)

        extra_data = {
            "mask_entire_sequence": mask_entire_sequence,
            "mask_single_date": mask_single_date,
            "batch_size": target_x.shape[0],
            "corr_features": corr_features,
        }

    return {
        "target_x": target_x,
        "target_y": target_y,
        "date_mask": date_mask,
        "dates": dates,
        "target_tickers": target_tickers,
        "vol_scaling_amount": vol_scaling_amount,
        "vol_scaling_amount_prev": vol_scaling_amount_prev,
        "trans_cost_bp": trans_cost_bp,
        "date_time_embedding_index": date_time_embedding_index,
        "to_weight_x": to_weight_x,
        **extra_data,
    }
