"""
This is a boilerplate pipeline 'data_encoding'
generated using Kedro 0.18.12
"""

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd
from typing import Tuple


# ------------------------------------------------------------------------------
# Transform categorical variables with few categories (class definition)
# ------------------------------------------------------------------------------
class FewCatEncoder(BaseEstimator):
    """An estimator designed to one-hot encode variables with few categories.
    It binarizes color, and pickup/dropoff boroughs, with respectively yellow
    and Manhattan as positive classes, and uses two fitted ordinal encoders on
    weekday and hour values after cutting the latter variable into smaller bins.
    """
    def __init__(self, weekday_enc: OneHotEncoder, hour_enc: OneHotEncoder):
        # Fitted encoders for weekday and hour 
        self.weekday_enc = weekday_enc
        self.hour_enc = hour_enc

    def transform(self, few: pd.DataFrame) -> pd.DataFrame:
        # Binarize color
        encoded_color = (few["color"] == "yellow").astype(int)
        encoded_color = pd.DataFrame(encoded_color)

        # Binarize pickup and dropoff boroughs
        M = "Manhattan"
        
        encoded_pickup_borough = (few["pickup_borough"] == M).astype(int)
        encoded_pickup_borough = pd.DataFrame(encoded_pickup_borough)

        encoded_dropoff_borough = (few["dropoff_borough"] == M).astype(int)
        encoded_dropoff_borough = pd.DataFrame(encoded_dropoff_borough)

        # Encode weekday
        encoded_weekday = self.weekday_enc.transform(few[["weekday"]])
        enc_weekday_cat = self.weekday_enc.categories_[0].astype("str")
        enc_weekday_lab = ["weekday_" + x for x in enc_weekday_cat]
        encoded_weekday = pd.DataFrame(encoded_weekday, columns=enc_weekday_lab)
        
        # Encode hour intervals
        encoded_hour = self.hour_enc.transform(pd.DataFrame(few[["hour"]]))
        enc_hour_cat = few["hour"].astype("category").cat.categories.astype("str")
        enc_hour_lab = ["hour_" + x for x in enc_hour_cat]
        encoded_hour = pd.DataFrame(encoded_hour, columns=enc_hour_lab)
        
        # Merge columns
        few_cat_features = pd.concat(
            [encoded_color, encoded_pickup_borough, encoded_dropoff_borough,
             encoded_weekday, encoded_hour], axis=1
        )

        # Rename
        cols = ["color", "pickup_borough", "dropoff_borough"]
        few_cat_features.rename(
            columns={f"{x}": f"encoded_{x}" for x in cols}, inplace=True
        )

        return few_cat_features


# ------------------------------------------------------------------------------
# Transform categorical variables with few categories (function)
# ------------------------------------------------------------------------------
def encode_few_cat(few_train: pd.DataFrame) -> Tuple:
    """encodedares categorical variables with few categories for the model.

    Args:
        few_train: Training variables with few categories.

    Returns:
        few_cat_encoder: Few categories variables' encoder.
    """
    # Weekday one-hot encoder
    weekday_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    weekday_enc.fit(few_train[["weekday"]])
    
    # Hour one-hot encoder
    hour_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    hour_enc.fit(few_train[["hour"]])

    # Full pipeline
    few_cat_encoder = FewCatEncoder(weekday_enc=weekday_enc, hour_enc=hour_enc)
    
    return few_cat_encoder


# ------------------------------------------------------------------------------
# Transform categorical variables with a lot of categories (class definition)
# ------------------------------------------------------------------------------
class LotCatEncoder(BaseEstimator):
    """An estimator designed to encode the variables with a lot of categories.
    """
    def __init__(
        self, pickup_zone_enc: OrdinalEncoder, dropoff_zone_enc: OrdinalEncoder
    ):
        # Fitted encoders for pickup and dropoff zones
        self.pickup_zone_enc = pickup_zone_enc
        self.dropoff_zone_enc = dropoff_zone_enc
    
    def transform(self, lot: pd.DataFrame) -> pd.DataFrame:
        # Encode pickup zone 
        encoded_pickup_zone = self.pickup_zone_enc.transform(
            lot[["pickup_zone"]]
        )
        encoded_pickup_zone = pd.DataFrame(
            encoded_pickup_zone, columns=["encoded_pickup_zone"]
        )
        
        # Encode dropoff zone
        encoded_dropoff_zone = self.dropoff_zone_enc.transform(
            lot[["dropoff_zone"]]
        )
        encoded_dropoff_zone = pd.DataFrame(
            encoded_dropoff_zone, columns=["encoded_dropoff_zone"]
        )

        # Merge columns
        lot_cat_features = pd.concat(
            [encoded_pickup_zone, encoded_dropoff_zone], axis=1
        )

        return lot_cat_features


# ------------------------------------------------------------------------------
# Transform categorical variables with few categories (function)
# ------------------------------------------------------------------------------
def encode_lot_cat(lot_train: pd.DataFrame) -> Tuple:
    """encodedares categorical variables with a lot of categories for the model.

    Args:
        lot_train: Training variables with a lot of categories.

    Returns:
        lot_cat_encoder: Lot of categories variables' encoder.
    """
    # Pickup zone label encoder
    pickup_zone_enc = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    pickup_zone_enc.fit(lot_train[["pickup_zone"]])

    # Dropoff zone label encoder
    dropoff_zone_enc = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    )
    dropoff_zone_enc.fit(lot_train[["dropoff_zone"]])
    
    # Full pipeline
    lot_cat_encoder = LotCatEncoder(
        pickup_zone_enc=pickup_zone_enc,
        dropoff_zone_enc=dropoff_zone_enc
    )
    
    return lot_cat_encoder


# ------------------------------------------------------------------------------
# Transform training and testing dependant variables
# ------------------------------------------------------------------------------
def encode_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Fit encoders to training dependant variables and transform both training
    and testing data accordingly.
    
    Args:
        X_train: Training dependant variables.
        X_test: Testing dependant variables.
    Returns:
        encoded_X_train: encodedrocessed training dependant variables.
        encoded_X_test: encodedrocessed testing dependant variables.
    """
    num = ["passengers", "distance", "trip_duration"]
    few = ["color", "pickup_borough", "dropoff_borough", "weekday", "hour"]
    lot = ["pickup_zone", "dropoff_zone"]

    # Fit the encoders to training data
    few_cat_encoder = encode_few_cat(few_train=X_train[few])
    lot_cat_encoder = encode_lot_cat(lot_train=X_train[lot])

    # Transformation training data
    few_train = few_cat_encoder.transform(X_train[few])
    lot_train = lot_cat_encoder.transform(X_train[lot])
    encoded_X_train = pd.concat([X_train[num], few_train, lot_train], axis=1)

    # Transform testing data
    few_test = few_cat_encoder.transform(X_test[few])
    lot_test = lot_cat_encoder.transform(X_test[lot])
    encoded_X_test = pd.concat([X_test[num], few_test, lot_test], axis=1)

    return encoded_X_train, encoded_X_test
