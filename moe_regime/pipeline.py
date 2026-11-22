"""End-to-end: data -> features -> regime labels -> purged walk-forward -> fit -> evaluate.

Every fold is evaluated twice, with and without the sentiment features, so the
`ablation` block answers the project question: does sentiment add anything over price?
"""
import json
import os

import numpy as np
import pandas as pd

from . import REGIMES
from .data import load_csv, make_synthetic
from .experts import RidgeExpert
from .features import build_features, next_day_return
from .labels import label_regimes
from .metrics import confusion, macro_f1, trading_stats
from .moe import RegimeMoE

VARIANTS = ("with_sentiment", "price_only")
SUMMARY_KEYS = {"router_acc": ("router_acc",), "router_macro_f1": ("router_macro_f1",),
                "moe_sharpe": ("moe", "sharpe"), "moe_gross_sharpe": ("moe", "gross_sharpe"),
                "moe_ic": ("moe", "ic"), "moe_turnover": ("moe", "turnover"),
                "single_sharpe": ("single_model_baseline", "sharpe"), "buy_and_hold_cum_ret": ("buy_and_hold_cum_ret",)}


def load_data(cfg: dict) -> pd.DataFrame:
    if cfg.get("csv"):
        return load_csv(cfg["csv"], cfg.get("sentiment_csv"), cfg.get("sentiment_lag", 1))
    return make_synthetic(cfg.get("n", 2500), cfg.get("seed", 0), cfg.get("sentiment_signal", 0.6))


def prepare(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    X = build_features(df)
    data = pd.concat([X, label_regimes(df, **cfg.get("label", {})), next_day_return(df)], axis=1).dropna()
    return data, list(X.columns)


def walk_forward_splits(n: int, n_folds: int, train_frac: float, horizon: int):
    """Expanding-window folds over the last (1 - train_frac) of the data.

    Fold i tests on block [s_i, e_i) and trains on [0, s_i - horizon): the `horizon` rows
    before each test block are purged because their forward-looking labels see test prices.
    """
    bounds = np.linspace(int(n * train_frac), n, n_folds + 1).astype(int)
    return [(slice(0, s - horizon), slice(s, e)) for s, e in zip(bounds[:-1], bounds[1:])]


def fit_eval(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str], cfg: dict):
    alpha, cost = cfg.get("expert_alpha", 10.0), cfg.get("cost_bps", 0.0)
    router_kw = {"seed": cfg.get("seed", 0), **(cfg.get("router") or {})}
    Xtr, Xte = tr[feats].to_numpy(), te[feats].to_numpy()
    moe = RegimeMoE(router_kw, alpha).fit(Xtr, tr["regime"].to_numpy(), tr["target"].to_numpy())
    pred, gate = moe.predict(Xte)
    hard = gate.argmax(1)
    bpred = RidgeExpert(alpha).fit(Xtr, tr["target"].to_numpy()).predict(Xte)
    reg, ret = te["regime"].to_numpy(), te["target"].to_numpy()
    out = {
        "test_start": str(te.index[0].date()), "test_end": str(te.index[-1].date()),
        "n_train": len(tr), "n_test": len(te),
        "router_acc": float((hard == reg).mean()),
        "router_macro_f1": macro_f1(reg, hard),
        "confusion(rows=true,cols=pred;" + ",".join(REGIMES) + ")": confusion(reg, hard).tolist(),
        "test_regime_share": {r: float((reg == i).mean()) for i, r in enumerate(REGIMES)},
        "moe": trading_stats(pred, ret, cost),
        "single_model_baseline": trading_stats(bpred, ret, cost),
        "buy_and_hold_cum_ret": float(np.exp(ret.sum()) - 1),
    }
    res = te[["regime", "target"]].copy()
    res["pred_regime"] = [REGIMES[i] for i in hard]
    for i, r in enumerate(REGIMES):
        res[f"p_{r}"] = gate[:, i]
    res["moe_pred"] = pred
    return out, res


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def summarize(folds: list[dict]) -> dict:
    return {k: {"mean": float(np.mean(v := [_get(f, p) for f in folds])), "std": float(np.std(v))}
            for k, p in SUMMARY_KEYS.items()}


def run(cfg: dict) -> dict:
    data, feats = prepare(load_data(cfg), cfg)
    h = cfg.get("label", {}).get("horizon", 20)
    splits = walk_forward_splits(len(data), cfg.get("n_folds", 5), cfg.get("train_frac", 0.7), h)
    feat_sets = {"with_sentiment": feats, "price_only": [f for f in feats if not f.startswith("sent")]}
    folds, routing = {v: [] for v in VARIANTS}, []
    for tr_sl, te_sl in splits:
        tr, te = data.iloc[tr_sl], data.iloc[te_sl]
        for v in VARIANTS:
            out, res = fit_eval(tr, te, feat_sets[v], cfg)
            folds[v].append(out)
            if v == "with_sentiment":
                routing.append(res)
    summary = {v: summarize(folds[v]) for v in VARIANTS}
    # paired per-fold differences: same train/test rows, only the feature set differs
    diffs = {k: [_get(a, p) - _get(b, p) for a, b in zip(folds["with_sentiment"], folds["price_only"])]
             for k, p in SUMMARY_KEYS.items() if k != "buy_and_hold_cum_ret"}
    out = {
        "n_rows": len(data), "n_folds": len(splits), "horizon": h, "cost_bps": cfg.get("cost_bps", 0.0),
        "summary": summary,
        "ablation_with_minus_price_only": {
            k: {"mean": float(np.mean(d)), "std": float(np.std(d)), "folds_positive": int(np.sum(np.array(d) > 0))}
            for k, d in diffs.items()},
        "folds": folds,
    }
    if cfg.get("out_dir"):
        os.makedirs(cfg["out_dir"], exist_ok=True)
        pd.concat(routing).to_csv(os.path.join(cfg["out_dir"], "routing_test.csv"))
        with open(os.path.join(cfg["out_dir"], "metrics.json"), "w") as fh:
            json.dump(out, fh, indent=2)
    return out
