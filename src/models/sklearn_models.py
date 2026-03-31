from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def create_sklearn_model(model_name: str, params: Dict[str, Any]) -> Any:
    """Factory for sklearn regressors."""
    if model_name == "linear_regression":
        return LinearRegression(**params)
    if model_name == "decision_tree":
        return DecisionTreeRegressor(**params)
    if model_name == "random_forest":
        return RandomForestRegressor(**params)
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    raise ValueError(f"Unsupported sklearn model: {model_name}")

