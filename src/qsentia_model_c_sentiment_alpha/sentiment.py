from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import feedparser
import numpy as np
import pandas as pd
import requests

from .model_c import robust_z_cross_section


COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
    "TSLA": "Tesla",
    "AVGO": "Broadcom",
    "BRK.B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase",
    "LLY": "Eli Lilly",
    "V": "Visa",
    "MA": "Mastercard",
    "NFLX": "Netflix",
    "COST": "Costco",
    "WMT": "Walmart",
    "HD": "Home Depot",
    "AMD": "AMD",
    "CRM": "Salesforce",
    "ORCL": "Oracle",
    "BAC": "Bank of America",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "DIS": "Disney",
    "ADBE": "Adobe",
    "CSCO": "Cisco",
    "TMO": "Thermo Fisher",
    "NKE": "Nike",
    "MCD": "McDonald's",
}

POSITIVE_WORDS = {
    "beat", "beats", "bullish", "upgrade", "upgraded", "surge", "rally", "record",
    "growth", "strong", "profit", "profitable", "optimistic", "raises", "raised",
    "outperform", "buy", "higher", "wins", "approval", "approved", "expands",
}
NEGATIVE_WORDS = {
    "miss", "misses", "bearish", "downgrade", "downgraded", "falls", "drop", "drops",
    "lawsuit", "probe", "fraud", "weak", "warning", "warns", "cut", "cuts", "layoff",
    "loss", "decline", "slump", "sell", "lower", "recall", "investigation",
}


@dataclass
class SourceDiagnostic:
    source: str
    symbol: str
    url: str
    http_status: int | None
    entries: int
    kept: int
    error: str | None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(entry) -> datetime:
    for key in ["published", "updated", "created"]:
        raw = entry.get(key)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return _now_utc()


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:900]


def _fetch_feed(url: str, timeout: int = 15) -> tuple[int | None, list, str | None]:
    try:
        headers = {"User-Agent": "QSentia research paper-trading bot/0.1"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        status = resp.status_code
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        return status, list(feed.entries), None
    except Exception as exc:
        return None, [], str(exc)[:240]


def _queries_for_symbol(symbol: str) -> list[tuple[str, str]]:
    clean = symbol.replace(".", "-")
    name = COMPANY_NAMES.get(symbol)
    queries = [
        ("yahoo", f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote_plus(clean)}&region=US&lang=en-US"),
    ]
    if name:
        q = quote_plus(f'"{name}" OR "{symbol}" stock earnings shares when:3d')
        queries.append(("google_news", f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"))
    else:
        q = quote_plus(f'"{symbol}" stock shares earnings when:3d')
        queries.append(("google_news", f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"))
    return queries


def choose_sentiment_symbols(base_score: pd.Series, max_symbols: int, per_side: int) -> list[str]:
    base_score = base_score.dropna().sort_values()
    bottom = base_score.head(per_side).index.tolist()
    top = base_score.tail(per_side).index.tolist()
    priority = [*reversed(top), *bottom]
    seen: set[str] = set()
    out: list[str] = []
    for sym in priority:
        if sym not in seen:
            out.append(sym)
            seen.add(sym)
        if len(out) >= max_symbols:
            break
    return out


def collect_live_text(symbols: list[str], lookback_hours: int, max_rows: int) -> tuple[pd.DataFrame, list[dict]]:
    cutoff = _now_utc() - timedelta(hours=lookback_hours)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    seen_ids: set[str] = set()

    for symbol in symbols:
        for source, url in _queries_for_symbol(symbol):
            status, entries, error = _fetch_feed(url)
            kept = 0
            for entry in entries:
                published_at = _parse_dt(entry)
                if published_at < cutoff:
                    continue
                title = _clean_text(entry.get("title", ""))
                summary = _clean_text(entry.get("summary", ""))
                if not title:
                    continue
                uid = hashlib.sha1(f"{symbol}|{title}|{entry.get('link', '')}".encode("utf-8")).hexdigest()
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                company = COMPANY_NAMES.get(symbol, symbol)
                prompt = (
                    f"Ticker: {symbol}. Company: {company}. "
                    f"Assess whether this fresh market news is positive, neutral, or negative for the stock. "
                    f"Headline: {title}. Summary: {summary}"
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "source": source,
                        "published_at": published_at.isoformat(),
                        "title": title,
                        "summary": summary,
                        "url": entry.get("link", ""),
                        "prompt": prompt,
                    }
                )
                kept += 1
                if len(rows) >= max_rows:
                    break
            diagnostics.append(SourceDiagnostic(source, symbol, url, status, len(entries), kept, error).__dict__)
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("published_at").drop_duplicates(["symbol", "title"]).reset_index(drop=True)
    return df, diagnostics


def _lexicon_scores(texts: list[str]) -> np.ndarray:
    scores = []
    for text in texts:
        words = set(re.findall(r"[a-zA-Z']+", text.lower()))
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        raw = (pos - neg) / max(1, pos + neg)
        scores.append(float(np.clip(raw, -1.0, 1.0)))
    return np.array(scores, dtype=float)


def _finbert_scores(texts: list[str], model_name: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            tok = tokenizer(batch, truncation=True, padding=True, max_length=192, return_tensors="pt")
            probs = torch.softmax(model(**tok).logits, dim=-1).cpu().numpy()
            for row in probs:
                pos = sum(row[i] for i, label in id2label.items() if "positive" in label)
                neg = sum(row[i] for i, label in id2label.items() if "negative" in label)
                scores.append(float(pos - neg))
    return np.array(scores, dtype=float)


def score_text_rows(df: pd.DataFrame, enable_transformer: bool, model_name: str) -> tuple[pd.DataFrame, str]:
    if df.empty:
        out = df.copy()
        out["sentiment_score"] = []
        return out, "none"
    texts = df["prompt"].astype(str).tolist()
    engine = "lexicon"
    if enable_transformer:
        try:
            scores = _finbert_scores(texts, model_name)
            engine = model_name
        except Exception as exc:
            print(f"Transformer sentiment unavailable; falling back to lexicon: {exc}")
            scores = _lexicon_scores(texts)
    else:
        scores = _lexicon_scores(texts)
    out = df.copy()
    out["sentiment_score"] = scores
    return out, engine


def aggregate_symbol_sentiment(scored: pd.DataFrame, index: pd.Index, min_symbol_rows: int) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(index=index, columns=["sentiment_raw", "sentiment_z", "sentiment_rows"]).fillna(0.0)
    agg = scored.groupby("symbol").agg(sentiment_raw=("sentiment_score", "mean"), sentiment_rows=("sentiment_score", "size"))
    agg = agg.reindex(index)
    raw = agg["sentiment_raw"].where(agg["sentiment_rows"].fillna(0) >= min_symbol_rows)
    z = robust_z_cross_section(raw)
    return pd.DataFrame(
        {
            "sentiment_raw": agg["sentiment_raw"].fillna(0.0),
            "sentiment_z": z.reindex(index).fillna(0.0),
            "sentiment_rows": agg["sentiment_rows"].fillna(0).astype(int),
        },
        index=index,
    )


def combine_scores(base_score: pd.Series, sentiment_frame: pd.DataFrame, alpha: float, min_total_rows: int) -> tuple[pd.Series, bool]:
    enough = int(sentiment_frame["sentiment_rows"].sum()) >= min_total_rows
    if not enough:
        return base_score.copy(), False
    combined = base_score.add(alpha * sentiment_frame["sentiment_z"], fill_value=0.0)
    return robust_z_cross_section(combined), True
