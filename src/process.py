"""Predict product purity, then choose setpoints under a cost budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution
from sklearn.ensemble import GradientBoostingRegressor

FEATURE_NAMES = ("temperature_c", "pressure_bar", "feed_rate")
BOUNDS = ((70.0, 110.0), (1.0, 8.0), (2.0, 10.0))
BUDGET = 11.0


def purity_true(x: np.ndarray) -> np.ndarray:
    temperature, pressure, feed = x[:, 0], x[:, 1], x[:, 2]
    temp_term = 0.18 * (temperature - 92) - 0.012 * (temperature - 92) ** 2
    pressure_term = 1.6 * pressure - 0.18 * pressure ** 2
    feed_term = 1.4 * feed - 0.16 * feed ** 2
    return 78.0 + temp_term + pressure_term + feed_term


def operating_cost(x: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(x)
    return 0.06 * values[:, 0] + 0.85 * values[:, 1] + 0.7 * values[:, 2]


def sample(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    draws = np.column_stack(
        [
            rng.uniform(BOUNDS[0][0], BOUNDS[0][1], n),
            rng.uniform(BOUNDS[1][0], BOUNDS[1][1], n),
            rng.uniform(BOUNDS[2][0], BOUNDS[2][1], n),
        ]
    )
    noise = rng.normal(0, 0.45, n)
    return draws, purity_true(draws) + noise


@dataclass
class Recommendation:
    setpoint: dict[str, float]
    predicted_purity: float
    cost: float
    feasible: bool


class ProcessModel:
    def __init__(self) -> None:
        self.regressor = GradientBoostingRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.08,
            random_state=7,
        )

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self.regressor.fit(features, target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.regressor.predict(np.atleast_2d(features))

    def _maximize(self, budget: float | None) -> np.ndarray:
        def objective(point: np.ndarray) -> float:
            return -float(self.predict(point)[0])

        kwargs = {"bounds": BOUNDS, "seed": 7, "polish": False}
        if budget is not None:
            kwargs["constraints"] = NonlinearConstraint(
                lambda point: float(operating_cost(point)[0]),
                -np.inf,
                budget,
            )
        return differential_evolution(objective, **kwargs).x

    def recommend(self, budget: float = BUDGET) -> Recommendation:
        point = self._maximize(budget)
        cost = float(operating_cost(point)[0])
        return Recommendation(
            setpoint={name: float(value) for name, value in zip(FEATURE_NAMES, point)},
            predicted_purity=float(self.predict(point)[0]),
            cost=cost,
            feasible=cost <= budget + 1e-6,
        )
