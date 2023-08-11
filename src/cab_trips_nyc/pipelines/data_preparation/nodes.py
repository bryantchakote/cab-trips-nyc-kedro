"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.18.12
"""

from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------------------
# Clean the raw data
# ------------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deletes some columns, removes lines with missing values in the dataset,
    computes trip duration, and retrieve weekday and hour from pickup.
    
    Args:
        df: Raw data.

    Returns:
        Preprocessed data.
    """
    # Delete some columns
    df.drop(["tip", "tolls", "total", "payment"], axis=1, inplace=True)

    # Remove NAs
    df.dropna(inplace=True)

    # Add trip duration
    df["trip_duration"] = (df["dropoff"] - df["pickup"]).dt.total_seconds()

    # Add (pickup) weekday
    df["weekday"] = df["pickup"].dt.strftime("%a")
    
    # Add (pickup) hour interval
    hour = df["pickup"].dt.hour
    bins = [0, 8, 17, 24]
    df["hour"] = pd.cut(hour, bins=bins, include_lowest=True, right=False)

    # Drop pickup and dropoff
    df.drop(["pickup", "dropoff"], axis=1, inplace=True)

    return df


# ------------------------------------------------------------------------------
# Divide the data into train and test sets
# ------------------------------------------------------------------------------
def split_data(df: pd.DataFrame, params: Dict) -> Tuple:
    """Splits the data in training and testing slices.

    Args:
        df: Cleaned data.
        params: Split parameters.

    Returns: 
        X_train: Training dependant variables.
        X_test: Testing dependant variables.
        y_train: Testing labels.
        y_test: Training labels.
    """
    # ISolate target variable
    X, y = df.drop("fare", axis=1), df[["fare"]]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )
    
    return X_train, X_test, y_train, y_test
