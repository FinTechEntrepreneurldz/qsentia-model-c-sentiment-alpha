# QSentia Model C Sentiment Alpha

Model C is the anchor: a sector-neutral 130/30 equity strategy using the existing QSentia MLP artifact trained on Ichimoku, chart-summary, and trailing momentum features.

This repo adds a live equity news sentiment overlay:

- Builds the same Model C daily feature frame and MLP score.
- Selects the strongest long and short candidates for fresh ticker-specific news.
- Pulls free RSS/news sources, including Yahoo Finance ticker feeds and Google News RSS.
- Scores headlines with FinBERT when the NLP extra is installed, with a deterministic lexicon fallback.
- Combines `base_score + SENTIMENT_ALPHA * sentiment_z`.
- Builds a 130/30 sector-neutral target book and paper trades through Alpaca.
- Writes dashboard-ready logs under `logs/`.

This is research/paper trading infrastructure, not production investment advice.

## Default Behavior

- Rebalance cadence: every 5 trading days by default.
- Execution: Alpaca paper account only.
- Universe: the symbols embedded in `artifacts/mlp.joblib`.
- Sentiment is conservative: if there are not enough fresh rows, the system falls back to base Model C unless `REQUIRE_LIVE_SENTIMENT=true`.

## GitHub Secrets

Set these in the repo:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`

## Useful Variables

| Variable | Default | Meaning |
| --- | ---: | --- |
| `SUBMIT_ORDERS` | `true` | Submit Alpaca paper orders when not in dry-run mode. |
| `REBALANCE_DAYS` | `5` | Approximate trading-day rebalance interval. |
| `SENTIMENT_ALPHA` | `0.35` | Strength of sentiment overlay relative to base Model C z-score. |
| `SENTIMENT_MAX_SYMBOLS` | `40` | Max candidate tickers to fetch news for each run. |
| `SENTIMENT_CANDIDATE_PER_SIDE` | `25` | Pull sentiment for top and bottom Model C candidates. |
| `SENTIMENT_MIN_TOTAL_ROWS` | `12` | Minimum scored news rows before sentiment affects the book. |
| `SENTIMENT_ENABLE_TRANSFORMER` | `true` | Use FinBERT if available; otherwise lexicon fallback. |
| `REQUIRE_LIVE_SENTIMENT` | `false` | Fail instead of falling back to base Model C when live text is thin. |

## Local Dry Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
pytest
python scripts/run_paper_trader.py --dry-run --force-rebalance
```

For FinBERT locally:

```bash
pip install -e ".[nlp,test]"
SENTIMENT_ENABLE_TRANSFORMER=true python scripts/run_paper_trader.py --dry-run --force-rebalance
```

## Manual Alpaca Paper Rebalance

```bash
ALPACA_API_KEY=... \
ALPACA_SECRET_KEY=... \
python scripts/run_paper_trader.py --force-rebalance
```

## Dashboard Logs

- `logs/portfolio/portfolio.csv`
- `logs/decisions/latest_decision.csv`
- `logs/target_weights/latest_target_weights.csv`
- `logs/orders/latest_planned_orders.csv`
- `logs/orders/latest_orders.csv`
- `logs/positions/latest_positions.csv`
- `logs/health/health_status.json`
- `logs/health/latest_sentiment_rows.csv`

## Suggested Dashboard Model Entry

```yaml
- id: qsentia_model_c_sentiment_alpha
  name: "QSentia Model C Sentiment Alpha"
  description: "Sector-neutral Model C equity MLP with live FinBERT news sentiment overlay and Alpaca paper execution."
  repo: "FinTechEntrepreneurldz/qsentia-model-c-sentiment-alpha"
  logs_path: "logs"
  branch: "main"
  enabled: true
  color: "#6366f1"
```
