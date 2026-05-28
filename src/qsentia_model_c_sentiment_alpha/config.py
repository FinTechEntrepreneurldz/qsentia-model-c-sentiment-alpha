from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "mlp.joblib"
LOG_DIR = REPO_ROOT / "logs"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in {None, ""}:
        return default
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in {None, ""}:
        return default
    return int(raw)


@dataclass(frozen=True)
class RuntimeConfig:
    data_period: str = os.environ.get("DATA_PERIOD", "2y")
    submit_orders: bool = env_bool("SUBMIT_ORDERS", True)
    default_account_value: float = env_float("DEFAULT_ACCOUNT_VALUE", 1_000_000.0)
    min_trade_notional: float = env_float("MIN_TRADE_NOTIONAL", 250.0)
    rebalance_days: int = env_int("REBALANCE_DAYS", 5)

    sentiment_alpha: float = env_float("SENTIMENT_ALPHA", 0.35)
    sentiment_max_symbols: int = env_int("SENTIMENT_MAX_SYMBOLS", 40)
    sentiment_per_side: int = env_int("SENTIMENT_CANDIDATE_PER_SIDE", 25)
    sentiment_max_rows: int = env_int("SENTIMENT_MAX_ROWS", 300)
    sentiment_min_total_rows: int = env_int("SENTIMENT_MIN_TOTAL_ROWS", 12)
    sentiment_min_symbol_rows: int = env_int("SENTIMENT_MIN_SYMBOL_ROWS", 1)
    sentiment_enable_transformer: bool = env_bool("SENTIMENT_ENABLE_TRANSFORMER", True)
    sentiment_require_live: bool = env_bool("REQUIRE_LIVE_SENTIMENT", False)
    sentiment_model: str = os.environ.get("SENTIMENT_MODEL", "ProsusAI/finbert")
    sentiment_lookback_hours: int = env_int("SENTIMENT_LOOKBACK_HOURS", 72)


def ensure_log_dirs() -> None:
    for sub in ["decisions", "orders", "positions", "portfolio", "target_weights", "health"]:
        (LOG_DIR / sub).mkdir(parents=True, exist_ok=True)
