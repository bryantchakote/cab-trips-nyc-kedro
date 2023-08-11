"""
This is a boilerplate pipeline 'data_encoding'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import encode_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=encode_data,
                inputs=["X_train", "X_test"],
                outputs=["encoded_X_train", "encoded_X_test"],
                name="data_encoder",
            ),
        ]
    )
