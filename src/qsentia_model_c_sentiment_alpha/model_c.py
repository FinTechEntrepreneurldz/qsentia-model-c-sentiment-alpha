from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

TRAILING_WINDOWS = [5, 21, 63, 126, 252]


def download_universe_prices(symbols: list[str], period: str) -> dict:
    raw = yf.download(
        symbols,
        period=period,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    def extract_field(raw_df: pd.DataFrame, field: str) -> pd.DataFrame:
        if isinstance(raw_df.columns, pd.MultiIndex):
            if field in raw_df.columns.get_level_values(1):
                return raw_df.xs(field, axis=1, level=1)
            if field in raw_df.columns.get_level_values(0):
                return raw_df.xs(field, axis=1, level=0)
        if field in raw_df.columns:
            return raw_df[[field]]
        return pd.DataFrame()

    close_all = extract_field(raw, "Close").sort_index().dropna(axis=1, how="all")
    high_all = extract_field(raw, "High").reindex_like(close_all)
    low_all = extract_field(raw, "Low").reindex_like(close_all)
    vol_all = extract_field(raw, "Volume").reindex_like(close_all)
    close_all = close_all.sort_index().ffill(limit=2)

    if close_all.empty:
        raise RuntimeError("yfinance returned no price data.")

    available = [s for s in symbols if s in close_all.columns]
    close = close_all[available]
    high = high_all.reindex(columns=available).reindex_like(close).ffill()
    low = low_all.reindex(columns=available).reindex_like(close).ffill()
    volume = vol_all.reindex(columns=available).reindex_like(close).fillna(0.0)

    return {
        "symbols": available,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "dollar_vol": close * volume,
    }


def _chart_summary_features(px_window: pd.Series) -> dict:
    x = pd.Series(px_window).astype(float).dropna()
    if len(x) < 10:
        return {f"f{i}": np.nan for i in range(8)}
    norm = x / x.iloc[0] - 1.0
    t = np.arange(len(norm))
    slope = np.polyfit(t, norm.values, 1)[0]
    quad = np.polyfit(t, norm.values, 2)[0]
    eq = (1 + x.pct_change().fillna(0)).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    dur = float(((eq / eq.cummax() - 1) < -0.02).sum()) / len(eq)
    total = (x.diff().abs()).sum()
    net = x.iloc[-1] - x.iloc[0]
    eff = float(net / total) if total > 0 else 0.0
    vol = float(x.pct_change().std() * np.sqrt(252)) if len(x) > 5 else 0.0
    rets = x.pct_change().dropna()
    resid = float(rets.tail(5).mean() - rets.mean()) if len(rets) > 5 else 0.0
    if len(norm) > 12:
        a1 = np.polyfit(np.arange(len(norm) // 3), norm.values[: len(norm) // 3], 1)[0]
        a2 = np.polyfit(np.arange(len(norm) // 3), norm.values[-(len(norm) // 3) :], 1)[0]
        accel = float(a2 - a1)
    else:
        accel = 0.0
    return dict(
        f0=float(slope),
        f1=float(quad),
        f2=float(dd),
        f3=float(dur),
        f4=float(eff),
        f5=float(vol),
        f6=float(resid),
        f7=float(accel),
    )


def build_chart_features_today(univ: dict, lookback_days: int = 63) -> pd.DataFrame:
    rows = []
    for sym in univ["symbols"]:
        px = univ["close"][sym].dropna()
        if len(px) < 10:
            continue
        win = px.tail(lookback_days)
        feats = {}
        for prefix in ["chart_price", "chart_rs_spy", "chart_rs_top"]:
            for k, v in _chart_summary_features(win).items():
                feats[f"{prefix}_{k}"] = v
        feats["symbol"] = sym
        rows.append(feats)
    return pd.DataFrame(rows).set_index("symbol")


def _compute_ichimoku(close_s, high_s, low_s, conv=9, base=26, span_b=52):
    tenkan = (
        high_s.rolling(conv, min_periods=max(3, conv // 2)).max()
        + low_s.rolling(conv, min_periods=max(3, conv // 2)).min()
    ) / 2
    kijun = (
        high_s.rolling(base, min_periods=max(5, base // 2)).max()
        + low_s.rolling(base, min_periods=max(5, base // 2)).min()
    ) / 2
    span_a = (tenkan + kijun) / 2
    span_b_line = (
        high_s.rolling(span_b, min_periods=max(10, span_b // 2)).max()
        + low_s.rolling(span_b, min_periods=max(10, span_b // 2)).min()
    ) / 2
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "span_a": span_a,
        "span_b": span_b_line,
        "cloud_top": pd.concat([span_a, span_b_line], axis=1).max(axis=1),
        "cloud_bot": pd.concat([span_a, span_b_line], axis=1).min(axis=1),
        "cloud_mid": pd.concat([span_a, span_b_line], axis=1).mean(axis=1),
    }


def _ichi_block_today(prefix: str, close_s: pd.Series, ichi: dict) -> dict:
    last = close_s.iloc[-1]
    out = {}
    out[f"{prefix}_vs_cloud_mid"] = float(last / ichi["cloud_mid"].iloc[-1] - 1.0) if ichi["cloud_mid"].iloc[-1] else np.nan
    out[f"{prefix}_above_cloud"] = float(last > ichi["cloud_top"].iloc[-1])
    out[f"{prefix}_below_cloud"] = float(last < ichi["cloud_bot"].iloc[-1])
    out[f"{prefix}_cloud_state"] = float(np.where(ichi["span_a"].iloc[-1] > ichi["span_b"].iloc[-1], 1.0, -1.0))
    out[f"{prefix}_tenkan_kijun_cross"] = float(np.where(ichi["tenkan"].iloc[-1] > ichi["kijun"].iloc[-1], 1.0, -1.0))
    out[f"{prefix}_tenkan_minus_kijun"] = float((ichi["tenkan"].iloc[-1] - ichi["kijun"].iloc[-1]) / last) if last else np.nan
    out[f"{prefix}_cloud_thickness"] = float(abs(ichi["cloud_top"].iloc[-1] - ichi["cloud_bot"].iloc[-1]) / last) if last else np.nan
    out[f"{prefix}_dist_to_span_a"] = float((last - ichi["span_a"].iloc[-1]) / last) if last else np.nan
    out[f"{prefix}_dist_to_span_b"] = float((last - ichi["span_b"].iloc[-1]) / last) if last else np.nan
    out[f"{prefix}_tenkan_slope_5"] = float(ichi["tenkan"].iloc[-1] / ichi["tenkan"].iloc[-6] - 1.0) if len(ichi["tenkan"]) > 6 and ichi["tenkan"].iloc[-6] else 0.0
    out[f"{prefix}_kijun_slope_5"] = float(ichi["kijun"].iloc[-1] / ichi["kijun"].iloc[-6] - 1.0) if len(ichi["kijun"]) > 6 and ichi["kijun"].iloc[-6] else 0.0
    return out


def build_ichimoku_features_today(univ: dict, conv=9, base=26, span_b=52) -> pd.DataFrame:
    rows = []
    for sym in univ["symbols"]:
        c = univ["close"][sym].dropna()
        if len(c) < span_b + 2:
            continue
        h = univ["high"][sym].reindex(c.index).ffill()
        l = univ["low"][sym].reindex(c.index).ffill()
        ichi_d = _compute_ichimoku(c, h, l, conv, base, span_b)
        feats = _ichi_block_today("ichi_d_price", c, ichi_d)
        feats.update(_ichi_block_today("ichi_d_rs_spy", c, ichi_d))
        feats.update(_ichi_block_today("ichi_d_rs_top", c, ichi_d))

        wc = c.resample("W-FRI").last().ffill()
        wh = h.resample("W-FRI").max().reindex(wc.index).ffill()
        wl = l.resample("W-FRI").min().reindex(wc.index).ffill()
        if len(wc) >= span_b + 2:
            ichi_w = _compute_ichimoku(wc, wh, wl, conv, base, span_b)
            feats.update(_ichi_block_today("ichi_w_price", wc, ichi_w))
            feats.update(_ichi_block_today("ichi_w_rs_spy", wc, ichi_w))
            feats.update(_ichi_block_today("ichi_w_rs_top", wc, ichi_w))
        feats["symbol"] = sym
        rows.append(feats)
    return pd.DataFrame(rows).set_index("symbol")


def build_trailing_features_today(univ: dict) -> pd.DataFrame:
    rows = []
    for sym in univ["symbols"]:
        px = univ["close"][sym].dropna()
        feats = {}
        for w in TRAILING_WINDOWS:
            tail = px.tail(w)
            if len(tail) >= max(5, w // 4):
                ret = float(tail.iloc[-1] / tail.iloc[0] - 1.0) if tail.iloc[0] > 0 else np.nan
                rets_d = tail.pct_change().dropna()
                vol = float(rets_d.std() * np.sqrt(252)) if len(rets_d) > 5 else np.nan
                eq = (1 + rets_d).cumprod()
                dd = float((eq / eq.cummax() - 1).min()) if len(eq) else np.nan
            else:
                ret = vol = dd = np.nan
            feats[f"ret_{w}d"] = ret
            feats[f"vol_{w}d"] = vol
            feats[f"dd_{w}d"] = dd
        for w in [21, 63]:
            feats[f"resid_{w}d"] = 0.0
        feats["symbol"] = sym
        rows.append(feats)
    return pd.DataFrame(rows).set_index("symbol")


def build_feature_frame_today(univ: dict, feature_specs: dict) -> pd.DataFrame:
    chart = build_chart_features_today(univ, lookback_days=feature_specs.get("chart_lookback_days", 63))
    ichi = build_ichimoku_features_today(
        univ,
        conv=feature_specs.get("ichi_tenkan", 9),
        base=feature_specs.get("ichi_kijun", 26),
        span_b=feature_specs.get("ichi_span_b", 52),
    )
    trail = build_trailing_features_today(univ)
    return chart.join(ichi, how="outer").join(trail, how="outer")


def robust_z_cross_section(x: pd.Series, clip: float = 4.0) -> pd.Series:
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    med = x.median()
    mad = (x - med).abs().median()
    if mad <= 1e-12 or pd.isna(mad):
        std = x.std()
        if std <= 1e-12 or pd.isna(std):
            return pd.Series(0.0, index=x.index)
        z = (x - x.mean()) / std
    else:
        z = (x - med) / (1.4826 * mad)
    return z.clip(-clip, clip).fillna(0.0)


def score_universe_mlp(features_df: pd.DataFrame, mlp_artifact: dict) -> pd.Series:
    features_df = features_df.copy()
    feature_cols = mlp_artifact["feature_cols"]
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = np.nan
    pred = mlp_artifact["model"].predict(features_df[feature_cols].astype(float).values)
    return robust_z_cross_section(pd.Series(pred, index=features_df.index))


def cap_and_renormalize(raw: pd.Series, mode: str, cap: float = 0.04) -> pd.Series:
    raw = pd.Series(raw, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if mode == "ls":
        pos_t, neg_t = 0.5, 0.5
    else:
        pos_t = raw[raw > 0].sum()
        neg_t = -raw[raw < 0].sum()
        net = pos_t - neg_t
        if abs(net) > 1e-12:
            raw = raw / net
            pos_t = raw[raw > 0].sum()
            neg_t = -raw[raw < 0].sum()
        else:
            pos_t, neg_t = 1.0, 0.0

    def _cap_amounts(amts: pd.Series, target: float, cap_: float) -> pd.Series:
        amts = pd.Series(amts, dtype=float).fillna(0.0)
        if target <= 0 or amts.empty:
            return amts * 0.0
        if amts.sum() <= 0:
            amts = pd.Series(target / len(amts), index=amts.index)
        else:
            amts = amts / amts.sum() * target
        for _ in range(25):
            over = amts > cap_ + 1e-12
            if not over.any():
                break
            fixed = amts[over].clip(upper=cap_)
            free = amts[~over]
            remaining = target - fixed.sum()
            if remaining <= 0 or free.empty:
                return pd.concat([fixed, free * 0.0]).reindex(amts.index).fillna(0.0)
            free = free / free.sum() * remaining if free.sum() > 0 else pd.Series(remaining / len(free), index=free.index)
            amts = pd.concat([fixed, free]).reindex(amts.index).fillna(0.0)
        return amts

    out = pd.Series(0.0, index=raw.index, dtype=float)
    if (raw > 0).any():
        out.loc[raw[raw > 0].index] = _cap_amounts(raw[raw > 0].abs(), pos_t, cap)
    if (raw < 0).any():
        out.loc[raw[raw < 0].index] = -_cap_amounts(raw[raw < 0].abs(), neg_t, cap)
    return out.fillna(0.0)


def make_weights(score: pd.Series, sectors: pd.Series, eligible: pd.Series, mode: str = "130_30", q: float = 0.20) -> pd.Series:
    score = pd.Series(score).where(eligible).replace([np.inf, -np.inf], np.nan)
    raw = pd.Series(0.0, index=score.index, dtype=float)
    n_total = max(1, score.dropna().shape[0])

    for _, idx in sectors.groupby(sectors).groups.items():
        cols = [c for c in idx if c in score.index]
        x = score.reindex(cols).dropna()
        if len(x) < 5 or x.nunique() < 2 or x.std() <= 1e-12:
            continue
        n = max(1, int(np.floor(len(x) * q)))
        top = x.nlargest(n).index
        bot = x.nsmallest(n).index
        sector_w = len(x) / n_total
        if mode == "130_30":
            raw.loc[top] += 1.30 * sector_w / n
            raw.loc[bot] -= 0.30 * sector_w / n
        elif mode == "long_top20":
            raw.loc[top] += sector_w / n
        elif mode == "long_exclude_bottom20":
            keep = x.drop(index=bot).index
            if len(keep):
                raw.loc[keep] += sector_w / len(keep)
        elif mode == "ls":
            raw.loc[top] += sector_w / n
            raw.loc[bot] -= sector_w / n
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return cap_and_renormalize(raw, mode)
