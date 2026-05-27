from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def get_alpaca_client():
    from alpaca.trading.client import TradingClient

    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")
    return TradingClient(api_key, api_secret, paper=True)


def get_account_info(client) -> dict:
    acct = client.get_account()
    return {
        "equity": float(acct.equity),
        "cash": float(acct.cash),
        "buying_power": float(acct.buying_power),
        "long_value": float(acct.long_market_value),
        "short_value": float(acct.short_market_value),
        "status": str(acct.status),
    }


def get_current_positions(client) -> pd.DataFrame:
    positions = client.get_all_positions()
    if not positions:
        return pd.DataFrame(columns=["symbol", "qty", "market_value", "side"])
    rows = []
    for pos in positions:
        qty = float(pos.qty)
        rows.append(
            {
                "symbol": pos.symbol,
                "qty": qty,
                "market_value": float(pos.market_value),
                "side": "long" if qty > 0 else "short",
            }
        )
    return pd.DataFrame(rows)


def build_order_plan(
    target_weights: pd.Series,
    current_positions: pd.DataFrame,
    account_value: float,
    prices: pd.Series,
    min_trade_notional: float,
) -> pd.DataFrame:
    target_weights = target_weights[target_weights.abs() > 1e-6]
    all_syms = sorted(set(target_weights.index.tolist()) | set(current_positions["symbol"].tolist() if not current_positions.empty else []))
    rows = []
    current_qty = pd.Series(dtype=float)
    if not current_positions.empty:
        current_qty = current_positions.set_index("symbol")["qty"].astype(float)

    for sym in all_syms:
        px = float(prices.get(sym, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        target_dollars = float(target_weights.get(sym, 0.0)) * account_value
        target_qty = int(np.trunc(target_dollars / px))
        held_qty = int(np.trunc(current_qty.get(sym, 0.0)))
        delta_qty = target_qty - held_qty
        if delta_qty == 0:
            continue
        notional = abs(delta_qty) * px
        if notional < min_trade_notional:
            continue
        rows.append(
            {
                "symbol": sym,
                "side": "buy" if delta_qty > 0 else "sell",
                "qty": int(abs(delta_qty)),
                "notional": float(notional),
                "price": px,
                "current_qty": held_qty,
                "target_qty": target_qty,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_rank"] = df["side"].map({"sell": 0, "buy": 1}).fillna(2)
    return df.sort_values(["_rank", "notional"], ascending=[True, False]).drop(columns="_rank").reset_index(drop=True)


def submit_orders(order_plan: pd.DataFrame, client) -> pd.DataFrame:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    submitted = []
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for i, row in order_plan.iterrows():
        try:
            req = MarketOrderRequest(
                symbol=row["symbol"],
                qty=int(row["qty"]),
                side=OrderSide.BUY if row["side"] == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=f"qsentia-mcsa-{run_id}-{i}-{row['symbol']}-{row['side']}-{int(row['qty'])}",
            )
            order = client.submit_order(req)
            submitted.append(
                {
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "qty": int(row["qty"]),
                    "notional": float(row["notional"]),
                    "order_id": str(order.id),
                    "status": str(order.status),
                }
            )
        except Exception as exc:
            submitted.append(
                {
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "qty": int(row["qty"]),
                    "notional": float(row["notional"]),
                    "order_id": "ERROR",
                    "status": str(exc)[:500],
                }
            )
    return pd.DataFrame(submitted)
