from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import LOG_DIR, ensure_log_dirs


def ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def append_csv(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def log_outputs(
    decision: dict,
    target_frame: pd.DataFrame,
    positions: pd.DataFrame,
    order_plan: pd.DataFrame,
    submitted: pd.DataFrame,
    account_info: dict,
    sentiment_rows: pd.DataFrame,
    diagnostics: list[dict],
) -> None:
    ensure_log_dirs()
    ts = ts_now()

    decision_row = pd.DataFrame(
        [
            {
                "timestamp_utc": ts,
                **decision,
                "equity": account_info["equity"],
                "cash": account_info["cash"],
            }
        ]
    )
    decision_row.to_csv(LOG_DIR / "decisions" / "latest_decision.csv", index=False)
    append_csv(LOG_DIR / "decisions" / "decisions.csv", decision_row)

    portfolio_row = pd.DataFrame(
        [
            {
                "timestamp_utc": ts,
                "equity": account_info["equity"],
                "cash": account_info["cash"],
                "long_value": account_info["long_value"],
                "short_value": account_info["short_value"],
                "buying_power": account_info["buying_power"],
                "n_positions": len(positions),
            }
        ]
    )
    append_csv(LOG_DIR / "portfolio" / "portfolio.csv", portfolio_row)

    tw = target_frame.copy()
    if "timestamp_utc" not in tw.columns:
        tw.insert(0, "timestamp_utc", ts)
    tw.to_csv(LOG_DIR / "target_weights" / "latest_target_weights.csv", index=False)

    pos = positions.copy()
    if pos.empty:
        pos = pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])
    pos.insert(0, "timestamp_utc", ts)
    pos.to_csv(LOG_DIR / "positions" / "latest_positions.csv", index=False)

    planned = order_plan.copy()
    if planned.empty:
        planned = pd.DataFrame(columns=["symbol", "side", "qty", "notional", "price", "current_qty", "target_qty"])
    planned.insert(0, "timestamp_utc", ts)
    planned.to_csv(LOG_DIR / "orders" / "latest_planned_orders.csv", index=False)

    sub = submitted.copy()
    if sub.empty:
        sub = pd.DataFrame(columns=["symbol", "side", "qty", "notional", "order_id", "status"])
    sub.insert(0, "timestamp_utc", ts)
    sub.to_csv(LOG_DIR / "orders" / "latest_orders.csv", index=False)
    if not submitted.empty:
        append_csv(LOG_DIR / "orders" / "orders.csv", sub)

    sent = sentiment_rows.copy()
    if sent.empty:
        sent = pd.DataFrame(columns=["symbol", "source", "published_at", "title", "summary", "url", "prompt", "sentiment_score"])
    sent.insert(0, "timestamp_utc", ts)
    sent.to_csv(LOG_DIR / "health" / "latest_sentiment_rows.csv", index=False)

    write_json(
        LOG_DIR / "health" / "health_status.json",
        {
            "timestamp_utc": ts,
            "decision": decision,
            "account": account_info,
            "sentiment_rows": int(len(sentiment_rows)),
            "sentiment_sources": sentiment_rows["source"].value_counts().to_dict() if not sentiment_rows.empty else {},
            "diagnostics": diagnostics[:100],
        },
    )


def get_last_rebalance_date() -> pd.Timestamp | None:
    history = LOG_DIR / "decisions" / "decisions.csv"
    latest = LOG_DIR / "decisions" / "latest_decision.csv"
    path = history if history.exists() else latest
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df = df[df.get("action", "") == "rebalance"]
        if df.empty:
            return None
        return pd.Timestamp(str(df["timestamp_utc"].iloc[-1])[:10])
    except Exception:
        return None


def is_rebalance_day(today: pd.Timestamp, rebalance_days: int, last_rebalance_date: pd.Timestamp | None) -> bool:
    if last_rebalance_date is None:
        return True
    days = (today - last_rebalance_date).days
    return days * 5 / 7 >= rebalance_days
