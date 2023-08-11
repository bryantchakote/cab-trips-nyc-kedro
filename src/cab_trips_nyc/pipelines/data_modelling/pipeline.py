"""
This is a boilerplate pipeline 'data_modelling'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["encoded_X_train", "y_train", "params:train_model_params"],
                outputs="random_forest_regressor",
                name="model_estimator"
            ),
            node(
                func=evaluate_model,
                inputs=["encoded_X_test", "y_test", "random_forest_regressor"],
                outputs="random_forest_metrics",
                name="model_evaluator"
            ),
        ]
    )
