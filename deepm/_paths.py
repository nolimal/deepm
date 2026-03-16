"""Canonical path resolution for the deepm package.

All paths are resolved relative to the project root (parent of the deepm/ package),
making the codebase independent of the current working directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
TRAIN_SETTINGS_DIR = CONFIGS_DIR / "train_settings"
BACKTEST_SETTINGS_DIR = CONFIGS_DIR / "backtest_settings"
SWEEP_SETTINGS_DIR = CONFIGS_DIR / "sweep_settings"
TCOST_DIR = CONFIGS_DIR / "tcost"

DATA_DIR = PROJECT_ROOT / "data"
DIAGNOSTICS_DIR = PROJECT_ROOT / "backtest_diagnostics"
