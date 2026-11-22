"""Data loading, real-data fetch helpers, and a synthetic regime-switching generator."""
import json
import urllib.request
import warnings

import numpy as np
import pandas as pd

from . import REGIMES

FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=1000&startTime={start}"


def load_csv(path: str, sentiment_csv: str | None = None, sentiment_lag: int = 1) -> pd.DataFrame:
    """Load `date, close[, volume, sentiment]`, optionally merging a separate `date, sentiment` CSV.

    Sentiment is shifted forward by `sentiment_lag` rows so the value used at date t was
    published before t's close (daily sentiment indices are typically stamped at 00:00 UTC
    but aggregate the whole day, so lag >= 1 is the safe default).
    """
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    if "close" not in df:
        raise ValueError("CSV must contain a 'close' column")
    if sentiment_csv:
        s = pd.read_csv(sentiment_csv, parse_dates=["date"]).set_index("date")["sentiment"]
        df = df.drop(columns="sentiment", errors="ignore").join(s, how="left")
    if "sentiment" not in df:
        warnings.warn("no sentiment column: sentiment features will be constant, ablation is meaningless")
        df["sentiment"] = 0.0
    df["sentiment"] = df["sentiment"].ffill().shift(sentiment_lag)
    return df


def parse_fear_greed(payload: dict) -> pd.DataFrame:
    """alternative.me Fear & Greed JSON -> `date, sentiment` scaled to [-1, 1]."""
    rows = [(pd.to_datetime(int(d["timestamp"]), unit="s").normalize(), (int(d["value"]) - 50) / 50)
            for d in payload["data"]]
    return pd.DataFrame(rows, columns=["date", "sentiment"]).sort_values("date").drop_duplicates("date")


def parse_binance_klines(rows: list) -> pd.DataFrame:
    """Binance /klines rows -> `date, close, volume` (open time, close, base volume)."""
    return pd.DataFrame({"date": pd.to_datetime([r[0] for r in rows], unit="ms").normalize(),
                         "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows]})


def _get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "moe-regime"}), timeout=30) as r:
        return json.load(r)


def fetch_btc_and_fear_greed(symbol: str = "BTCUSDT", start: str = "2018-02-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download daily BTC closes (Binance) and the Crypto Fear & Greed Index (alternative.me)."""
    t, end, parts = int(pd.Timestamp(start).timestamp() * 1000), pd.Timestamp.now("UTC").value // 10**6, []
    while t < end:
        rows = _get_json(BINANCE_KLINES.format(symbol=symbol, start=t))
        if not rows:
            break
        parts.append(parse_binance_klines(rows))
        t = rows[-1][0] + 86_400_000
    prices = pd.concat(parts).drop_duplicates("date").iloc[:-1]  # drop today's unfinished candle
    return prices, parse_fear_greed(_get_json(FNG_URL))


def make_synthetic(n: int = 2500, seed: int = 0, sentiment_signal: float = 0.6) -> pd.DataFrame:
    """Markov-switching price plus a sentiment series.

    `sentiment_signal` scales how much sentiment co-moves with the hidden regime drift.
    The default (0.6) plants a real signal; 0.0 gives pure-noise sentiment, which is the
    null case for the sentiment ablation.
    """
    rng = np.random.default_rng(seed)
    drift = {0: -0.0045, 1: 0.0, 2: 0.0050}
    vol = {0: 0.030, 1: 0.012, 2: 0.020}
    stay = 0.985
    state, states = 1, []
    for _ in range(n):
        if rng.random() > stay:
            state = int(rng.choice([s for s in range(3) if s != state]))
        states.append(state)
    states = np.array(states)
    ret = np.array([rng.normal(drift[s], vol[s]) for s in states])
    close = 100 * np.exp(np.cumsum(ret))
    raw = np.array([drift[s] / 0.005 for s in states]) * sentiment_signal
    sent = pd.Series(raw + rng.normal(0, 0.5, n)).ewm(span=5).mean().to_numpy()
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"close": close, "volume": rng.lognormal(10, 0.3, n), "sentiment": sent,
         "true_regime": [REGIMES[s] for s in states]}, index=idx)
