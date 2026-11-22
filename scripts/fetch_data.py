"""Download daily BTC closes (Binance) and the Crypto Fear & Greed Index (alternative.me).

Usage: python scripts/fetch_data.py [--out data] [--symbol BTCUSDT] [--start 2018-02-01]
Then:  python run_pipeline.py --csv data/btc.csv --sentiment-csv data/fear_greed.csv
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moe_regime.data import fetch_btc_and_fear_greed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--symbol", default="BTCUSDT")
ap.add_argument("--start", default="2018-02-01")  # Fear & Greed history starts 2018-02-01
a = ap.parse_args()
prices, fng = fetch_btc_and_fear_greed(a.symbol, a.start)
os.makedirs(a.out, exist_ok=True)
prices.to_csv(os.path.join(a.out, "btc.csv"), index=False)
fng.to_csv(os.path.join(a.out, "fear_greed.csv"), index=False)
print(f"wrote {len(prices)} price rows ({prices.date.min().date()}..{prices.date.max().date()}) and {len(fng)} sentiment rows to {a.out}/")
