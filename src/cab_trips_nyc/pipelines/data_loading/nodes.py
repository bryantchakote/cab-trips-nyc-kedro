"""
This is a boilerplate pipeline 'data_loading'
generated using Kedro 0.18.12
"""

import pandas as pd
import seaborn as sns


# ------------------------------------------------------------------------------
# Load the raw data
# ------------------------------------------------------------------------------
def load_data() -> pd.DataFrame():
    """Load the raw data.
    
    Args:
        None.
    
    Returns:
        df: Data summarizing cab trips in New York City in March 2019.
    """
    df = sns.load_dataset('taxis')
    return df
