import numpy as np


def confusion(y, p, k=3):
    m = np.zeros((k, k), int)
    for a, b in zip(y.astype(int), p.astype(int)):
        m[a, b] += 1
    return m


def macro_f1(y, p, k=3):
    m, f = confusion(y, p, k), []
    for i in range(k):
        tp = m[i, i]; pr = tp / max(m[:, i].sum(), 1); rc = tp / max(m[i].sum(), 1)
        f.append(0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


def trading_stats(pred, ret, cost_bps: float = 0.0):
    """Long/short by sign of prediction on daily log returns `ret`; crypto 365d/yr.

    P&L uses simple returns (so shorts are exact) minus `cost_bps` per unit of position
    change; a full long->short flip costs 2x. Reports both net and gross Sharpe.
    """
    pos = np.sign(pred)
    turnover = np.abs(np.diff(pos, prepend=0.0))
    gross = pos * np.expm1(ret)
    net = gross - turnover * cost_bps / 1e4
    ann = np.sqrt(365)
    ic = np.corrcoef(pred, ret)[0, 1] if pred.std() > 0 and ret.std() > 0 else 0.0
    return {"dir_acc": float((pos == np.sign(ret)).mean()), "ic": float(ic),
            "sharpe": float(net.mean() / (net.std() + 1e-12) * ann),
            "gross_sharpe": float(gross.mean() / (gross.std() + 1e-12) * ann),
            "turnover": float(turnover.mean()), "cum_ret": float(np.prod(1 + net) - 1)}
