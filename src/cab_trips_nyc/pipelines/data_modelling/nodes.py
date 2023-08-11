"""
This is a boilerplate pipeline 'data_modelling'
generated using Kedro 0.18.12
"""

from typing import Dict, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score


# ------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------
def train_model(encoded_X_train: pd.DataFrame, y_train: pd.Series, params: Dict):
    """Fit a random forest regressor to the training data.

    Args:
        X_train: Training features.
        y_train: Training target.
    Returns:
        reg: The trained model.
    """
    reg = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    reg.fit(encoded_X_train, y_train.values.ravel())
    
    return reg


# ------------------------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------------------------
def evaluate_model(
    encoded_X_test: pd.DataFrame, y_test: pd.Series, reg:RandomForestRegressor
) -> Tuple:
    """Evaluate the model's performance on the testing data.

    Args:
        X_test: Testing features.
        y_test: Testing target.
        reg: The trained model
    Returns:
        reg: The trained model.
    """
    # Infererence on testing data
    y_pred = reg.predict(encoded_X_test)

    # Metrics
    reg_r2 = r2_score(y_test, y_pred)
    reg_mse = mse(y_test, y_pred)
    metrics = pd.DataFrame({"R2": [reg_r2], "MSE": [reg_mse]})

    return metrics
