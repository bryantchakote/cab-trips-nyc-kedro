"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import clean_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs="raw_data",
                outputs="cleaned_data",
                name="data_cleaner",
            ),
            node(
                func=split_data,
                inputs=["cleaned_data", "params:train_test_split_params"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="data_spliter",
            ),
        ]
    )
