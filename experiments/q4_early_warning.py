"""Q4: can you see it coming? Early warning of the 5% index circuit breaker

Data: random redemption runs (prediction/runs.py), 30% of them calm. Every 5 steps from
day 7, features from public market data (prediction/features.py) and the label
"the 5% index breaker fires within the next 100 steps (~50 minutes)".

Evaluation, as for a fraud or credit model:
    - split BY RUN: test runs (seed ending in 7, 8, 9) are never seen in training.
      Splitting individual steps at random would leak: neighbouring steps of the same
      run are nearly identical
    - precision-recall (the event is rare: accuracy would be meaningless)
    - calibration: when the model says 30%, does it happen 30% of the time?
    - early warning: alarm threshold set on the training runs (recall 80%), then on
      the test runs: breakers caught, minutes of warning, false alarms
    - against a simple rule: "alarm when the index is far below yesterday's close"
    - two information sets: public market data only (any market participant), and
      public data + the fund's flows (what a regulator or the fund platform sees)

Usage:
    python experiments/q4_early_warning.py              # 300 runs (~7 min on 8 cores)
    python experiments/q4_early_warning.py --runs 60    # quicker
    python experiments/q4_early_warning.py --reuse      # skip the simulations, reuse the dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from contagionsim.prediction import (
    FEATURES,
    FLOW_FEATURES,
    draw_scenario,
    fund_flow_features,
    run_features,
    run_labels,
    sample_steps,
    simulate,
)
from contagionsim.scenarios import MiniBistConfig
from contagionsim.sweep import run_sweep

HORIZON = 100  # steps ahead (~50 minutes)
START = 7000  # runs start on day 7 or 8
EVERY = 5
WARN_WINDOW = 300  # an alarm up to 300 steps before a breaker counts as a warning
STEPS_PER_MINUTE = 2


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    scenario = draw_scenario(cell["seed"])
    sim, fund = simulate(scenario)
    ex = sim.exchange
    halts = [h for m in ex.markets.values() for h in m.halts]
    breakers = [t for t, th in ex.index_halts if th == 0.05]
    limit = MiniBistConfig().concentrated_rules.daily_limit
    X = run_features(sim.history, halts, fund.params.concentrated, fund.params.liquid, limit)
    X = np.hstack([X, fund_flow_features(fund)])
    y = run_labels(len(sim.history), breakers, HORIZON)
    steps = sample_steps(sim.history, breakers, START, EVERY)
    return {
        "seed": cell["seed"],
        "X": X[steps],
        "y": y[steps],
        "steps": steps,
        "breakers": breakers,
        "calm": scenario.is_calm,
    }


def build_dataset(n_runs: int, workers: int | None) -> dict[str, np.ndarray]:
    runs = run_sweep(run_cell, [{"seed": s} for s in range(n_runs)], workers=workers)
    return {
        "X": np.vstack([r["X"] for r in runs]),
        "y": np.concatenate([r["y"] for r in runs]),
        "seed": np.concatenate([np.full(len(r["y"]), r["seed"]) for r in runs]),
        "step": np.concatenate([r["steps"] for r in runs]),
        "breaker_seed": np.array([r["seed"] for r in runs for _ in r["breakers"]], dtype=int),
        "breaker_step": np.array([b for r in runs for b in r["breakers"]], dtype=int),
        "calm_seeds": np.array([r["seed"] for r in runs if r["calm"]], dtype=int),
    }


def is_test(seed: np.ndarray) -> np.ndarray:
    return np.isin(seed % 10, (7, 8, 9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("results/q4_dataset.npz"))
    ap.add_argument("--out", type=Path, default=Path("results/q4_early_warning.png"))
    args = ap.parse_args()

    if args.reuse:
        data = dict(np.load(args.data))
    else:
        print(f"simulating {args.runs} random runs ...")
        data = build_dataset(args.runs, args.workers)
        args.data.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.data, **data)
        print(f"dataset saved to {args.data}")

    X, y, seed, step = data["X"], data["y"].astype(bool), data["seed"], data["step"]
    test = is_test(seed)
    n_runs = len(np.unique(seed))
    n_breaker_runs = len(np.unique(data["breaker_seed"]))
    print(f"\n{n_runs} runs ({len(data['calm_seeds'])} calm), {n_breaker_runs} with a 5% breaker; "
          f"{len(y):,} samples, {y.mean():.2%} positive")  # fmt: skip
    print(f"train: {int((~test).sum()):,} samples from {len(np.unique(seed[~test]))} runs; "
          f"test: {int(test.sum()):,} samples from {len(np.unique(seed[test]))} runs")  # fmt: skip

    names = FEATURES + FLOW_FEATURES
    col = {f: i for i, f in enumerate(names)}
    public = np.arange(len(FEATURES))
    everything = np.arange(len(names))

    def logistic() -> Any:
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

    def boosting() -> Any:
        # Heavily regularized: there are only a few dozen breaker events to learn from
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=200,
            l2_regularization=1.0, random_state=0,
        )  # fmt: skip

    models: dict[str, tuple[Any, np.ndarray]] = {
        "logistic, public": (logistic(), public),
        "boosting, public": (boosting(), public),
        "logistic, + fund flows": (logistic(), everything),
        "boosting, + fund flows": (boosting(), everything),
    }
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {
        # The simple rule: the further the index is below yesterday's close, the closer the breaker
        "rule: index vs prev close": (
            -X[~test, col["index_ret_day"]],
            -X[test, col["index_ret_day"]],
        )
    }
    for name, (model, cols) in models.items():
        model.fit(X[~test][:, cols], y[~test])
        scores[name] = (
            model.predict_proba(X[~test][:, cols])[:, 1],
            model.predict_proba(X[test][:, cols])[:, 1],
        )

    results = {}
    print(f"\ntest set: base rate {y[test].mean():.2%}")
    print(f"{'model':<28}{'PR-AUC':>8}{'90% CI':>15}{'caught':>9}{'warning (min)':>15}"
          f"{'false alarms':>14}")  # fmt: skip
    for name, (s_train, s_test) in scores.items():
        ap_score = average_precision_score(y[test], s_test)
        lo, hi = bootstrap_ap(y[test], s_test, seed[test])
        threshold = threshold_for_recall(y[~test], s_train, 0.8)
        warn = early_warning(s_test >= threshold, seed[test], step[test], data)
        results[name] = {"ap": ap_score, "ci": (lo, hi), "scores": s_test, **warn}
        print(f"{name:<28}{ap_score:>8.3f}{f'{lo:.3f}-{hi:.3f}':>15}{warn['caught']:>9}"
              f"{warn['median_lead_min']:>15.0f}{warn['false_alarm_rate']:>13.1%}")  # fmt: skip
    for a, b in (("boosting, + fund flows", "boosting, public"),
                 ("boosting, + fund flows", "rule: index vs prev close")):  # fmt: skip
        d, lo, hi = bootstrap_ap_difference(
            y[test], results[a]["scores"], results[b]["scores"], seed[test]
        )
        print(f"PR-AUC gain, {a} over {b}: {d:+.3f} (paired 90% CI {lo:+.3f} to {hi:+.3f})")
    base_brier = brier_score_loss(y[test], np.full(int(test.sum()), y[~test].mean()))
    for name in models:
        brier = brier_score_loss(y[test], results[name]["scores"])
        print(
            f"brier score {name}: {brier:.4f} (always predicting the base rate: {base_brier:.4f})"
        )

    best = max(models, key=lambda m: results[m]["ap"])
    model, cols = models[best]
    imp = permutation_importance(
        model, X[test][:, cols], y[test], scoring="average_precision", n_repeats=5,
        random_state=0,
    )  # fmt: skip
    feature_names = [names[i] for i in cols]
    order = np.argsort(imp.importances_mean)[::-1]
    print(f"\n{best}: drop in test PR-AUC when a feature is shuffled")
    for i in order[:8]:
        print(f"  {feature_names[i]:<26}{imp.importances_mean[i]:>8.3f}")

    plot(y[test], results, imp, feature_names, best, args.out)


def bootstrap_ap(
    y: np.ndarray, s: np.ndarray, seed: np.ndarray, n: int = 1000
) -> tuple[float, float]:
    """90% interval of PR-AUC, resampling whole test RUNS (steps of a run are not independent)"""
    rng = np.random.default_rng(0)
    runs = np.unique(seed)
    rows = {r: np.flatnonzero(seed == r) for r in runs}
    values = []
    for _ in range(n):
        pick = np.concatenate([rows[r] for r in rng.choice(runs, size=len(runs))])
        if y[pick].any():
            values.append(average_precision_score(y[pick], s[pick]))
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def bootstrap_ap_difference(
    y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray, seed: np.ndarray, n: int = 1000
) -> tuple[float, float, float]:
    """PR-AUC(a) - PR-AUC(b), with a 90% interval from the SAME resampled runs for both

    Pairing removes the run-to-run noise that both models share, so it is much more
    sensitive than comparing two separate intervals
    """
    rng = np.random.default_rng(1)
    runs = np.unique(seed)
    rows = {r: np.flatnonzero(seed == r) for r in runs}
    diffs = []
    for _ in range(n):
        pick = np.concatenate([rows[r] for r in rng.choice(runs, size=len(runs))])
        if y[pick].any():
            diffs.append(
                average_precision_score(y[pick], s_a[pick])
                - average_precision_score(y[pick], s_b[pick])
            )
    point = average_precision_score(y, s_a) - average_precision_score(y, s_b)
    return float(point), float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))


def threshold_for_recall(y: np.ndarray, s: np.ndarray, recall: float) -> float:
    """Highest score threshold that still catches `recall` of the positives (training data)"""
    pos = np.sort(s[y])
    return float(pos[int((1 - recall) * len(pos))]) if len(pos) else float("inf")


def early_warning(
    alarm: np.ndarray, seed: np.ndarray, step: np.ndarray, data: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Breakers caught with warning, median warning time, and false-alarm rate

    caught: an alarm in the WARN_WINDOW steps before the breaker
    lead: time from the first such alarm to the breaker
    false alarms: share of samples far (> WARN_WINDOW) from any breaker that alarm
    """
    leads = []
    events = [
        (s, b)
        for s, b in zip(data["breaker_seed"], data["breaker_step"], strict=True)
        if is_test(np.array([s]))[0]
    ]
    near = np.zeros(len(step), dtype=bool)
    for s, b in events:
        window = (seed == s) & (step >= b - WARN_WINDOW) & (step < b)
        near |= (seed == s) & (step >= b - WARN_WINDOW) & (step <= b + WARN_WINDOW)
        hits = step[window & alarm]
        if len(hits):
            leads.append(b - hits.min())
    far = ~near
    return {
        "caught": f"{len(leads)}/{len(events)}",
        "median_lead_min": float(np.median(leads)) / STEPS_PER_MINUTE if leads else 0.0,
        "false_alarm_rate": float(alarm[far].mean()) if far.any() else 0.0,
    }


def plot(
    y: np.ndarray,
    results: dict[str, Any],
    imp: Any,
    feature_names: list[str],
    best: str,
    out: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping figure (pip install -e ".[plot]")')
        return

    ink, muted = "#1f1f1f", "#8a8a8a"
    colors = {
        "rule: index vs prev close": "#a3a3a3",
        "logistic, public": "#eb6834",
        "boosting, public": "#2a78d6",
        "logistic, + fund flows": "#1baf7a",
        "boosting, + fund flows": "#e87ba4",
    }
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": muted,
            "xtick.color": muted,
            "ytick.color": muted,
            "font.size": 9,
        }
    )
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

    for name, r in results.items():
        p, rc, _ = precision_recall_curve(y, r["scores"])
        ax[0].plot(rc, p, color=colors[name], lw=1.5, label=f"{name} (AP {r['ap']:.2f})")
    ax[0].axhline(y.mean(), color=muted, lw=0.8, ls="--")
    ax[0].annotate("base rate", (0.98, y.mean()), xytext=(-50, 4), textcoords="offset points",
                   color=muted, fontsize=8)  # fmt: skip
    ax[0].set(title="Precision-recall, test runs", xlabel="recall", ylabel="precision",
              xlim=(0, 1), ylim=(0, 0.5))  # fmt: skip
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")

    bins = np.linspace(0, 1, 11)
    for name in dict.fromkeys(("logistic, public", best)):
        s = results[name]["scores"]
        idx = np.digitize(s, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 20:
                xs.append(s[m].mean())
                ys.append(y[m].mean())
        ax[1].plot(xs, ys, "o-", color=colors[name], lw=1.2, ms=4, label=name)
    ax[1].plot([0, 0.5], [0, 0.5], color=muted, lw=0.8, ls="--")
    ax[1].set(title="Calibration, test runs (bins with 20+ samples)",
              xlabel="predicted probability", ylabel="observed frequency",
              xlim=(0, 0.5), ylim=(0, 0.5))  # fmt: skip
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")

    order = np.argsort(imp.importances_mean)[::-1][:8][::-1]
    ax[2].barh([feature_names[i] for i in order], imp.importances_mean[order],
               xerr=imp.importances_std[order], color=colors[best], height=0.6,
               error_kw={"ecolor": muted, "elinewidth": 1})  # fmt: skip
    ax[2].set(title=f"Which features matter ({best})", xlabel="drop in PR-AUC when shuffled")

    fig.suptitle("Q4: early warning of the 5% index circuit breaker within 50 minutes, "
                 "public market data vs + fund flows (regulator view)", color=ink)  # fmt: skip
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
