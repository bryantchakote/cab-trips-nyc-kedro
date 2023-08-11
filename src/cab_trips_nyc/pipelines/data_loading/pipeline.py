"""
This is a boilerplate pipeline 'data_loading'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs=None,
                outputs="raw_data",
                name="data_loader",
            ),
        ]
    )
