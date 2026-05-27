from __future__ import annotations

import argparse

import joblib
import pandas as pd

from .alpaca_client import build_order_plan, get_account_info, get_alpaca_client, get_current_positions, submit_orders
from .config import ARTIFACT_PATH, RuntimeConfig, ensure_log_dirs
from .logging_utils import get_last_rebalance_date, is_rebalance_day, log_outputs
from .model_c import build_feature_frame_today, download_universe_prices, make_weights, score_universe_mlp
from .sentiment import (
    aggregate_symbol_sentiment,
    choose_sentiment_symbols,
    collect_live_text,
    combine_scores,
    score_text_rows,
)


def run_cycle(force_rebalance: bool = False, dry_run: bool = False) -> dict:
    ensure_log_dirs()
    cfg = RuntimeConfig()
    artifact = joblib.load(ARTIFACT_PATH)
    symbols = artifact["symbols"]
    sectors = pd.Series(artifact["sectors"]).reindex(symbols).fillna("Unknown")

    print(f"Downloading Model C universe prices: {len(symbols)} symbols")
    univ = download_universe_prices(symbols, cfg.data_period)
    features = build_feature_frame_today(univ, artifact.get("feature_specs", {}))
    base_score = score_universe_mlp(features, artifact)

    sentiment_symbols = choose_sentiment_symbols(base_score, cfg.sentiment_max_symbols, cfg.sentiment_per_side)
    print(f"Collecting live sentiment for {len(sentiment_symbols)} candidates")
    text_rows, diagnostics = collect_live_text(sentiment_symbols, cfg.sentiment_lookback_hours, cfg.sentiment_max_rows)
    scored_text, sentiment_engine = score_text_rows(text_rows, cfg.sentiment_enable_transformer, cfg.sentiment_model)
    sentiment_frame = aggregate_symbol_sentiment(scored_text, base_score.index, cfg.sentiment_min_symbol_rows)
    combined_score, sentiment_used = combine_scores(base_score, sentiment_frame, cfg.sentiment_alpha, cfg.sentiment_min_total_rows)

    if cfg.sentiment_require_live and not sentiment_used:
        raise RuntimeError(
            f"Live sentiment gate failed: got {len(scored_text)} rows, "
            f"need at least {cfg.sentiment_min_total_rows}. "
            "Set REQUIRE_LIVE_SENTIMENT=false for diagnostics or base-model-only fallback."
        )

    eligible = pd.Series(True, index=base_score.index)
    mode = artifact.get("best_mode", "130_30")
    target_weights = make_weights(combined_score, sectors.reindex(base_score.index).fillna("Unknown"), eligible, mode=mode, q=artifact.get("tail_q", 0.20))
    prices = univ["close"].iloc[-1].reindex(base_score.index)

    target_frame = pd.DataFrame(
        {
            "symbol": target_weights.index,
            "weight": target_weights.values,
            "base_score": base_score.reindex(target_weights.index).values,
            "sentiment_raw": sentiment_frame.reindex(target_weights.index)["sentiment_raw"].values,
            "sentiment_z": sentiment_frame.reindex(target_weights.index)["sentiment_z"].values,
            "sentiment_rows": sentiment_frame.reindex(target_weights.index)["sentiment_rows"].values,
            "combined_score": combined_score.reindex(target_weights.index).values,
        }
    )
    target_frame = target_frame[target_frame["weight"].abs() > 1e-8].sort_values("weight", ascending=False).reset_index(drop=True)

    if dry_run:
        client = None
        account_info = {
            "equity": cfg.default_account_value,
            "cash": cfg.default_account_value,
            "buying_power": cfg.default_account_value * 2.0,
            "long_value": 0.0,
            "short_value": 0.0,
            "status": "dry_run",
        }
        positions = pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])
    else:
        client = get_alpaca_client()
        account_info = get_account_info(client)
        positions = get_current_positions(client)

    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    do_rebalance = force_rebalance or is_rebalance_day(today, cfg.rebalance_days, get_last_rebalance_date())
    order_plan = pd.DataFrame()
    submitted = pd.DataFrame()

    if do_rebalance:
        order_plan = build_order_plan(
            target_weights,
            positions,
            account_info["equity"],
            prices,
            min_trade_notional=cfg.min_trade_notional,
        )
        if not dry_run and cfg.submit_orders and not order_plan.empty:
            submitted = submit_orders(order_plan, client)
        action = "rebalance"
    else:
        action = "hold"

    decision = {
        "action": action,
        "do_rebalance": bool(do_rebalance),
        "submit_orders": bool((not dry_run) and cfg.submit_orders),
        "sentiment_used": bool(sentiment_used),
        "sentiment_engine": sentiment_engine,
        "sentiment_rows": int(len(scored_text)),
        "sentiment_symbols": int(len(sentiment_symbols)),
        "n_target_positions": int((target_weights.abs() > 1e-8).sum()) if do_rebalance else 0,
        "n_planned_orders": int(len(order_plan)),
        "n_submitted_orders": int(len(submitted)),
        "gross_exposure": float(target_weights.abs().sum()) if do_rebalance else 0.0,
        "long_exposure": float(target_weights[target_weights > 0].sum()) if do_rebalance else 0.0,
        "short_exposure": float(target_weights[target_weights < 0].sum()) if do_rebalance else 0.0,
        "status": "ok",
    }
    log_outputs(decision, target_frame, positions, order_plan, submitted, account_info, scored_text, diagnostics)
    print(decision)
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebalance", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_cycle(force_rebalance=args.force_rebalance, dry_run=args.dry_run)
    return 0
