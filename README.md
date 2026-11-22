# Impact of sentiment on the crypto market — MoE regime router (bull / sideways / bear)

**Research question:** does market sentiment add predictive information about crypto regimes and
next-day returns *beyond what price already contains*?

The model is a Mixture-of-Experts: a **router (gating network)** reads causal price + sentiment
features and outputs soft probabilities over three regimes; one **ridge expert** per regime predicts
the next-day return; output = Σ gate_e · expert_e. The question is answered by an **ablation**: every
walk-forward fold is fit twice on identical rows, once with and once without the `sent*` features.

```
data -> features (causal) -> regime labels (forward return / vol) -> purged walk-forward folds
     -> {with_sentiment, price_only} x [Router (softmax MLP) + 3 ridge experts] -> metrics + ablation
```

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py                                  # synthetic data
python scripts/fetch_data.py                            # BTC (Binance) + Fear & Greed (alternative.me) -> data/
python run_pipeline.py --csv data/btc.csv --sentiment-csv data/fear_greed.csv
python -m pytest
```
Add `--full` to print per-fold results. Outputs: `outputs/metrics.json` (summary, ablation, folds) and
`outputs/routing_test.csv` (router probabilities over all test folds).

## Data
- `--csv`: `date, close[, volume, sentiment]`. `--sentiment-csv`: optional `date, sentiment`, left-joined on date.
- Sentiment is forward-filled then **lagged `sentiment_lag` rows (default 1)** so the value used at day *t*
  was published before *t*'s close. Daily indices such as Fear & Greed summarise a whole UTC day, so lag 0 leaks.
- Fear & Greed (0–100) is rescaled to [-1, 1]. Its history starts 2018-02-01.
- Without `--csv`, a synthetic Markov-switching series is used. Its sentiment is built from the hidden
  regime drift with strength `sentiment_signal` (0.6 default; **0.0 = pure noise**, the null case), so
  synthetic results only check that the machinery works. They say nothing about real markets.

## Evaluation
- **Walk-forward:** the last `1 - train_frac` of rows is cut into `n_folds` contiguous test blocks; each fold
  trains on all earlier rows minus a `horizon`-row purge (regime labels look `horizon` days ahead).
  Metrics are reported as mean ± std across folds.
- **Trading stats:** long/short on the sign of the prediction, simple returns, `cost_bps` charged per unit of
  position change (a flip pays 2×). Net and gross Sharpe and turnover are both reported, alongside a
  single-ridge baseline and buy & hold.
- **Ablation:** `ablation_with_minus_price_only` gives paired per-fold differences (mean, std, and how many folds were positive).

## Results so far (synthetic only)
Default config, seed 0, 5 folds, 10 bps:

| | router acc | MoE IC | MoE net Sharpe |
|---|---|---|---|
| with sentiment (lag 0, i.e. in-memory synthetic) | 0.49 | 0.062 | 1.75 |
| price only | 0.45 | 0.047 | 1.35 |

Over 4 seeds, the ablation's IC gain is ~+0.04 with planted sentiment and ~0.00 with `sentiment_signal: 0`
(both are covered by tests). With the 1-day lag applied through the CSV path, the router-accuracy gain
remains (+0.04) but the Sharpe gain disappears, so **timing matters as much as content**.
Fold-to-fold std is large (router acc ±0.12), so no edge is established; the next step is running on real BTC + sentiment data.

Deps: numpy, pandas, pyyaml (pytest for tests). Swap `router.py`/`experts.py` for torch/sklearn models if desired.
