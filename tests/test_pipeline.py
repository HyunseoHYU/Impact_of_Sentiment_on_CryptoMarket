import os

import numpy as np
import pandas as pd
import pytest
import yaml

from moe_regime.data import load_csv, make_synthetic, parse_binance_klines, parse_fear_greed
from moe_regime.features import build_features
from moe_regime.labels import label_regimes
from moe_regime.metrics import trading_stats
from moe_regime.pipeline import run, walk_forward_splits
from moe_regime.router import Router

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


@pytest.fixture(scope="module")
def cfg():
    with open(CFG_PATH) as fh:
        c = yaml.safe_load(fh)
    c["out_dir"] = None
    return c


def test_features_are_causal():
    df = make_synthetic(400)
    f1 = build_features(df)
    df2 = df.copy(); df2.iloc[300:, df2.columns.get_loc("close")] *= 5
    f2 = build_features(df2)
    assert np.allclose(f1.iloc[:300].fillna(0), f2.iloc[:300].fillna(0))


def test_labels_cover_three_regimes():
    lab = label_regimes(make_synthetic(2000)).dropna()
    assert set(lab.unique()) == {0.0, 1.0, 2.0}


def test_router_topk_and_simplex():
    X = np.random.default_rng(0).normal(size=(200, 5)); y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1)
    p = Router(top_k=2, epochs=50).fit(X, y).predict_proba(X)
    assert np.allclose(p.sum(1), 1) and ((p > 0).sum(1) <= 2).all()


def test_walk_forward_folds_are_purged_and_disjoint():
    h, splits = 30, walk_forward_splits(1000, 5, 0.7, 30)
    tests = [range(1000)[te] for _, te in splits]
    assert tests[0].start == 700 and tests[-1].stop == 1000
    assert all(a.stop == b.start for a, b in zip(tests, tests[1:]))  # contiguous, no overlap
    for tr, te in splits:
        assert tr.start == 0 and tr.stop == te.start - h               # purge gap = horizon


def test_costs_reduce_sharpe_monotonically():
    rng = np.random.default_rng(0)
    pred, ret = rng.normal(size=500), rng.normal(0.001, 0.02, 500)
    s = [trading_stats(pred, ret, c)["sharpe"] for c in (0, 10, 50)]
    assert s[0] > s[1] > s[2]
    assert trading_stats(pred, ret, 0)["sharpe"] == trading_stats(pred, ret, 0)["gross_sharpe"]


def test_load_csv_lags_and_merges_sentiment(tmp_path):
    d = pd.date_range("2024-01-01", periods=4, freq="D")
    pd.DataFrame({"date": d, "close": [1, 2, 3, 4]}).to_csv(tmp_path / "p.csv", index=False)
    pd.DataFrame({"date": d, "sentiment": [0.1, 0.2, 0.3, 0.4]}).to_csv(tmp_path / "s.csv", index=False)
    df = load_csv(str(tmp_path / "p.csv"), str(tmp_path / "s.csv"), sentiment_lag=1)
    assert np.isnan(df["sentiment"].iloc[0]) and df["sentiment"].iloc[1:].tolist() == [0.1, 0.2, 0.3]


def test_parsers():
    fng = parse_fear_greed({"data": [{"value": "75", "timestamp": "1704153600"},
                                     {"value": "25", "timestamp": "1704067200"}]})
    assert fng["sentiment"].tolist() == [-0.5, 0.5] and fng["date"].iloc[0] == pd.Timestamp("2024-01-01")
    px = parse_binance_klines([[1704067200000, "1", "2", "0.5", "1.5", "10", 0, "0", 0, "0", "0", "0"]])
    assert px.iloc[0].tolist() == [pd.Timestamp("2024-01-01"), 1.5, 10.0]


def test_pipeline_beats_chance(cfg):
    accs = [run({**cfg, "seed": s})["summary"]["with_sentiment"]["router_acc"]["mean"] for s in range(4)]
    assert np.mean(accs) > 0.40  # 3 classes -> chance ~0.33; walk-forward mean is ~0.48


def test_ablation_detects_planted_sentiment_but_not_noise(cfg):
    def d_ic(signal):
        return np.mean([run({**cfg, "seed": s, "sentiment_signal": signal})
                        ["ablation_with_minus_price_only"]["moe_ic"]["mean"] for s in range(4)])
    assert d_ic(0.6) > d_ic(0.0) + 0.015
