"""Fit a purity model and print the best setpoint inside the cost budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from process import BUDGET, BOUNDS, ProcessModel, operating_cost, sample  # noqa: E402

REPORT = ROOT / "models" / "metrics.json"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(7)
    features, target = sample(800, rng)
    split = 640
    model = ProcessModel()
    model.fit(features[:split], target[:split])
    predicted = model.predict(features[split:])
    heldout = target[split:]
    baseline = np.array([[90.0, 4.0, 6.0]])
    free_point = model._maximize(None)
    recommendation = model.recommend(BUDGET)
    report = {
        "mae": float(mean_absolute_error(heldout, predicted)),
        "r2": float(r2_score(heldout, predicted)),
        "budget": BUDGET,
        "bounds": {name: list(bound) for name, bound in zip(("temperature_c", "pressure_bar", "feed_rate"), BOUNDS)},
        "baseline_setpoint": {"temperature_c": 90.0, "pressure_bar": 4.0, "feed_rate": 6.0},
        "baseline_predicted_purity": float(model.predict(baseline)[0]),
        "baseline_cost": float(operating_cost(baseline)[0]),
        "baseline_feasible": bool(operating_cost(baseline)[0] <= BUDGET),
        "unconstrained_setpoint": {
            name: float(value) for name, value in zip(("temperature_c", "pressure_bar", "feed_rate"), free_point)
        },
        "unconstrained_predicted_purity": float(model.predict(free_point)[0]),
        "unconstrained_cost": float(operating_cost(free_point)[0]),
        "optimized_setpoint": recommendation.setpoint,
        "optimized_predicted_purity": recommendation.predicted_purity,
        "optimized_cost": recommendation.cost,
        "feasible": recommendation.feasible,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test MAE: {report['mae']:.3f}")
    print(f"test R2:  {report['r2']:.3f}")
    print(
        f"unconstrained purity {report['unconstrained_predicted_purity']:.2f} "
        f"at cost {report['unconstrained_cost']:.2f}"
    )
    print(
        f"baseline purity {report['baseline_predicted_purity']:.2f} "
        f"at cost {report['baseline_cost']:.2f} feasible={report['baseline_feasible']}"
    )
    print(
        f"optimized purity {report['optimized_predicted_purity']:.2f} "
        f"at cost {report['optimized_cost']:.2f} (budget {BUDGET})"
    )
    print("setpoint:", recommendation.setpoint)


if __name__ == "__main__":
    main()
