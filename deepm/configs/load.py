import yaml

from deepm._paths import TRAIN_SETTINGS_DIR, SWEEP_SETTINGS_DIR

CORRELATION_SPAN_DEFAULT = 252

ARCHITECTURES = [
    "LSTM",
    "LSTM_SIMPLE",
    "MOM_TRANS",
    "AdvancedTemporalBaseline",
    "DeePM",
]


def load_train_settings(file_name: str) -> dict:
    """Load the train settings from YAML."""
    with open(
        TRAIN_SETTINGS_DIR / f"{file_name}.yaml",
        "r",
        encoding="UTF-8",
    ) as f:
        configs = yaml.safe_load(f)
    configs["description"] = file_name
    return configs


def load_settings_for_architecture(file_name: str, architecture: str) -> dict:
    """Load settings and apply architecture-specific defaults."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. Must be one of {ARCHITECTURES}")
    settings = load_train_settings(file_name)

    run_name = architecture
    if settings["test_run"]:
        run_name = f"TEST/{run_name}"
    settings["run_name"] = run_name

    settings["use_contexts"] = False
    settings["cross_section"] = architecture in ["DeePM"]
    settings["num_context"] = 0

    settings.setdefault("correlation_span", CORRELATION_SPAN_DEFAULT)
    settings.setdefault("local_time_embedding", False)
    settings.setdefault("train_target_override", None)
    settings.setdefault("valid_target_override", None)
    settings.setdefault("extra_data_pre_steps", 0)
    settings.setdefault("tcost_inputs", False)

    return settings


def load_sweep_settings(file_name: str) -> dict:
    """Load the sweep settings from YAML."""
    with open(
        SWEEP_SETTINGS_DIR / f"{file_name}.yaml",
        "r",
        encoding="UTF-8",
    ) as f:
        configs = yaml.safe_load(f)
    return configs
