"""Usage: python run_pipeline.py [--config configs/default.yaml] [--csv prices.csv] [--sentiment-csv s.csv]"""
import argparse, json, yaml
from moe_regime.pipeline import run

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--csv", default=None, help="CSV with date, close[, volume, sentiment]; omit for synthetic data")
ap.add_argument("--sentiment-csv", default=None, help="optional CSV with date, sentiment merged onto --csv")
ap.add_argument("--full", action="store_true", help="print per-fold results too")
a = ap.parse_args()
with open(a.config) as fh:
    cfg = yaml.safe_load(fh)
if a.csv:
    cfg["csv"] = a.csv
if a.sentiment_csv:
    cfg["sentiment_csv"] = a.sentiment_csv
out = run(cfg)
if not a.full:
    out.pop("folds")
print(json.dumps(out, indent=2))
