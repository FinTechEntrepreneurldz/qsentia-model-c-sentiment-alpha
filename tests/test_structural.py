from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from qsentia_model_c_sentiment_alpha.alpaca_client import build_order_plan
from qsentia_model_c_sentiment_alpha.config import ARTIFACT_PATH
from qsentia_model_c_sentiment_alpha.model_c import make_weights, robust_z_cross_section, score_universe_mlp
from qsentia_model_c_sentiment_alpha.sentiment import aggregate_symbol_sentiment, combine_scores, score_text_rows


def test_artifact_loads():
    artifact = joblib.load(ARTIFACT_PATH)
    expected = {"model", "feature_cols", "best_mode", "symbols", "sectors"}
    assert expected <= set(artifact.keys())
    assert artifact["best_mode"] == "130_30"
    assert len(artifact["symbols"]) > 100
    assert Path(ARTIFACT_PATH).exists()


def test_mlp_scores_synthetic_features():
    artifact = joblib.load(ARTIFACT_PATH)
    rng = np.random.default_rng(42)
    syms = artifact["symbols"][:80]
    features = pd.DataFrame(
        rng.normal(size=(len(syms), len(artifact["feature_cols"]))),
        index=syms,
        columns=artifact["feature_cols"],
    )
    score = score_universe_mlp(features, artifact)
    assert score.notna().all()
    assert len(score) == len(syms)


def test_sentiment_lexicon_and_overlay():
    base = pd.Series({"AAPL": 1.0, "MSFT": 0.4, "TSLA": -1.0, "JPM": -0.3})
    text = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "TSLA", "JPM"],
            "source": ["test"] * 4,
            "prompt": [
                "Apple beats estimates and raises guidance",
                "Apple shares rally after strong growth",
                "Tesla faces downgrade after weak deliveries",
                "JPMorgan neutral market update",
            ],
        }
    )
    scored, engine = score_text_rows(text, enable_transformer=False, model_name="unused")
    assert engine == "lexicon"
    frame = aggregate_symbol_sentiment(scored, base.index, min_symbol_rows=1)
    combined, used = combine_scores(base, frame, alpha=0.5, min_total_rows=3)
    assert used
    assert combined["AAPL"] > robust_z_cross_section(base)["AAPL"]
    assert combined.notna().all()


def test_make_weights_and_order_plan_share_delta():
    artifact = joblib.load(ARTIFACT_PATH)
    syms = artifact["symbols"]
    rng = np.random.default_rng(7)
    score = pd.Series(rng.normal(size=len(syms)), index=syms)
    sectors = pd.Series(artifact["sectors"]).reindex(syms).fillna("Unknown")
    weights = make_weights(score, sectors, pd.Series(True, index=syms), mode="130_30", q=0.20)
    assert 1.45 < weights.abs().sum() < 1.75
    assert 0.90 < weights.sum() < 1.10

    prices = pd.Series(100.0, index=weights.index)
    current = pd.DataFrame(
        {
            "symbol": [weights.index[0], weights.index[1], "XYZ"],
            "qty": [10, -5, 8],
            "market_value": [1000.0, -500.0, 800.0],
            "side": ["long", "short", "long"],
        }
    )
    plan = build_order_plan(weights, current, account_value=100_000.0, prices=prices, min_trade_notional=1.0)
    assert set(plan["side"]).issubset({"buy", "sell"})
    assert (plan["qty"] > 0).all()
    xyz = plan[plan["symbol"] == "XYZ"]
    assert not xyz.empty
    assert int(xyz.iloc[0]["qty"]) == 8
