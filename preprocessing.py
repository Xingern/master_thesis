import numpy as np
import pandas as pd

def convert_to_float_or_nan(value):
    """
    Function to convert a single value to float64, or NaN if it can't be 
    converted
    """
    try:
        return np.float64(value)
    except (ValueError, TypeError):
        return np.nan

def initial_clean(raw_df):
    """
    The functions does some basic pre-processing of the imported data. 
    It applies the following methods:
        1. Formats the time column to DateTime.
        2. Replaces all "Bad input" or similar values with NaN
        3. Downcasts all data to float32 for memory optimization
    """
    raw_df["Time"] = pd.to_datetime(raw_df["Time"]).dt.strftime('%Y-%m-%d %H:%M')
    raw_df.iloc[:, 1:] = raw_df.iloc[:, 1:].applymap(convert_to_float_or_nan)
    raw_df.iloc[:, 1:] = raw_df.iloc[:, 1:].astype('float32')
    return raw_df.copy()

def drop_nan_neg(df):
    """
    Returns a df with NaN and negative values removed
    """
    drop_nan_df = df.dropna()
    clean_df = drop_nan_df[(drop_nan_df.iloc[:, 1:] > 0).all(1)]
    return clean_df

