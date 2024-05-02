import os
import numpy as np
import pandas as pd

# These columns are badly formatted and will be dropped by default
DROP_COLS = ["Desorber-Packing-TT1030A", 
             "Downstream-Lean-PT1010", 
             "Desorber-Packing-TT1030C"]

STABLE = ["F1", "D1", "T2", "P6", "T7", "F8", "D8", "T9", "F10", "T10"]
STD_LIMIT = [0.03, 10, 1, 0.03, 0.5, 0.03, 10, 0.5, 5, 0.5]

STABLE = {'F1': 0.03, 'D1': 10, 'T2': 1, 'P6': 0.03, 'T7': 0.5, 'L7': 50, 
          'F8': 0.03, 'D8': 10, 'T9': 0.5, 'F10': 5, 'T10': 0.5, 'B11': 0.2, 
          'B12': 0.1, 'Cap': 1}

STABILITY_LEN = len(STABLE)

def lysekil(data_path):
    headers = ["Time", "Downstream-Rich-FT1042", "Downstream-Rich-AT1060", 
            "Downstream-Rich-TT1043", "Upstream-Rich-TT1027", 
            "Upstream-Rich-PST1025", "Desorber-Overhead-FT1067", 
            "Desorber-Overhead-PIC1032", "Desorber-Overhead-TIC1062", 
            "Desorber-Overhead-Corrected", "Desorber-Packing-PT1004", 
            "Desorber-Packing-TT1004", "Desorber-Packing-PT1005", 
            "Desorber-Packing-TT1005", "Desorber-Packing-TT1030A", 
            "Desorber-Packing-TT1030B", "Desorber-Packing-TT1030C", 
            "Desorber-Packing-PT1035", "Desorber-Sump-TT1029", 
            "Desorber-Sump-UX1029", "Desorber-Sump-TIC1029", 
            "Desorber-Sump-Corrected", "Upstream-Lean-TT1053", 
            "Downstream-Lean-FT1076", "Downstream-Lean-AT1058", 
            "Downstream-Lean-PT1010", "Downstream-Lean-TT1054"]

    # Keep the indices of columns with actual values
    all_cols = list(range(36))
    remove_cols = [0, 1, 3, 7, 10, 15, 24, 29, 31]
    import_cols = [x for x in all_cols if x not in remove_cols]

    # Rows to skip due to formating
    skip = 3

    data_file_path = os.path.join(data_path, "Lysekil_MEA_data_unprotected.xlsx")
    cache_file_path = os.path.join(data_path, "cached_Lysekil.pkl")

    # Check if the pickled file exists and load it
    if os.path.exists(cache_file_path):
        raw_df = pd.read_pickle(cache_file_path)
    else:
        raw_df = pd.read_excel(data_file_path, 
                               header=None, 
                               names=headers, 
                               usecols=import_cols, 
                               skiprows=skip)
        raw_df.to_pickle(cache_file_path)
        
    return raw_df

def extended_lysekil(data_path):
    headers = ['Time', 'Desorber-Sump-LT1017', 'Absorber-AE1062A_1', 
               'Absorber-AE1062A_2', 'Absorber-AE1062B_1', 
               'Absorber-AE1062B_2','Capture_rate']
    
    all_cols = list(range(10))
    remove_cols = [0, 1, 2]
    import_cols = [x for x in all_cols if x not in remove_cols]
    
    # Rows to skip due to formating
    skip = 3
    
    # Paths to data file and cache
    data_file_path = os.path.join(data_path, "Extended_Lysekil_MEA_data_fixed.xlsx")
    cache_file_path = os.path.join(data_path, "cached_Extended_Lysekil.pkl")

    # Check if the pickled file exists and load it
    if os.path.exists(cache_file_path):
        raw_df = pd.read_pickle(cache_file_path)
    else:
        raw_df = pd.read_excel(data_file_path, 
                               header=None, 
                               names=headers, 
                               usecols=import_cols, 
                               skiprows=skip)
        raw_df.to_pickle(cache_file_path)
        
    return raw_df

def initial_clean(raw_df, drop_cols=DROP_COLS):
    """
    The functions does some basic pre-processing of the imported data. 
    It applies the following methods:
        1. Formats the time column to DateTime.
        2. Replaces all "Bad input" or similar values with NaN
        3. Downcasts all data to float32 for memory optimization
        4. Drops selected columns
        5. Removes rows with NaN or negative values
    """
    
    raw_df["Time"] = pd.to_datetime(raw_df["Time"]).dt.strftime('%Y-%m-%d %H:%M')
    raw_df.iloc[:, 1:] = raw_df.iloc[:, 1:].map(convert_to_float_or_nan)
    raw_df[raw_df.columns[1:]] = raw_df[raw_df.columns[1:]].astype('float32')
    drop_df = raw_df.drop(columns=drop_cols)
    df = drop_nan_neg(drop_df)
    return df.copy()

def convert_to_float_or_nan(value):
    """
    Function to convert a single value to float64, or NaN if it can't be 
    converted
    """
    try:
        return np.float64(value)
    except (ValueError, TypeError):
        return np.nan

def drop_nan_neg(df):
    """
    Returns a df with NaN and negative values removed
    """
    drop_nan_df = df.dropna()
    clean_df = drop_nan_df[(drop_nan_df.iloc[:, 1:] > 0).all(1)]
    return clean_df

def find_closest_value(temp, pres, table):
    """
    Find the closest value in a table to the given temperature and pressure.
    
    Parameters
    ----------
    temp : float
        The temperature to find the closest value to.
    pres : float
        The pressure to find the closest value to.
    table : DataFrame
        The table to search in.
        
    Returns
    -------
    float
        The closest value in the table to the given temperature and pressure.
    """
    temp_vals = np.array([int(col[:-2]) for col in table.columns])
    closest_temp = temp_vals[np.abs(temp_vals - temp).argmin()]
    temp_col = f"{closest_temp}°C"
    closest_pres = table.index[np.abs(table.index - pres).argmin()]
    
    return table.at[closest_pres, temp_col]

def correct_CO2_flow(df, table):
    """
    Correct the CO2 flow in the dataframe using the given table.
    
    Parameters
    ----------
    df : DataFrame
        The dataframe to correct.
    table : DataFrame
        The table to use for correction.
        
    Returns
    -------
    DataFrame
        Dataframe with corrected CO2 flow.
    """
    df = df.copy()
    for index, row in df.iterrows():
        T, P, F = row["T10"], row["P6"], row["F10"]
        x_CO2 = find_closest_value(T, P, table)
        df.at[index, "F10"] = x_CO2 * F
    return df

def stability(df, start_date, end_date, stable=STABLE):
    """
    Calculate stability of a dataframe within a specified date range.

    Parameters
    ----------
    df (pandas.DataFrame): 
        The input dataframe.
        
    start_date (str): 
        The start date of the date range.
        
    end_date (str): 
        The end date of the date range.
        
     stable (dict): 
        A dictionary with keys as column names and values as standard limits.

    Returns
    -------
    tuple: A tuple containing two dataframes:
        - df_stable (pandas.DataFrame): A dataframe containing the actual deviation, limits, and status of stability.
        - df_p (pandas.DataFrame): A dataframe containing the subset of the input dataframe within the specified date range.
    """
    mask = (df["Time"] >= start_date) & (df["Time"] <= end_date)
    df_p = df.loc[mask]
    std = df_p[stable.keys()].std()
    df_stable = pd.DataFrame({'Actual deviation': np.round(std, 3)})
                              
    df_stable['Limits'] = df_stable.index.map(stable)

    df_stable['Status'] = np.where(df_stable['Actual deviation'] 
                                   <= df_stable['Limits'], 
                                   'Stable', 
                                   'Unstable')

    return df_stable, df_p

def time_frame_future(df, current_time, tol):
    """
    Returns the time that is current minute + X minutes in the future.
    If such time does not exist, takes as far as possible and uses previous 
    time for the remaining X minutes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'Time' column.
    current_time : str
        The current timestamp in the format 'YYYY-MM-DD HH:MM'.
    tol : int
        Number of minutes to look into the future.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame within the time frame.
        
    See Also
    --------
    time_frame_future_past() for both future and past time frames.
    """
    future_time = pd.to_datetime(current_time) + pd.Timedelta(minutes=tol)
    future_mask = (df['Time'] > current_time) & (df['Time'] <= future_time)
    
    if len(df[future_mask]) == tol:
        return df[future_mask]
    else:
        past_time_delta = tol - len(df[future_mask])
        past_time = pd.to_datetime(current_time) - pd.Timedelta(minutes=past_time_delta)
        past_mask = (df['Time'] > past_time) & (df['Time'] <= future_time)

        if len(df[past_mask]) != tol:
            return None
        else:
            return df[past_mask]
           
def time_frame_future_past(df, current_time, tol):
    """
    Extract a time frame from the past and future of a specified time point.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'Time' column.
    current_time : str
        The current timestamp in the format 'YYYY-MM-DD HH:MM'.
    tol : int
        Number of minutes to look into the future and past.
    
    Returns
    -------
    df : pandas.DataFrame
        DataFrame within the time frame.
        
    See Also
    --------
    time_frame_future() for only the future time frame.
    """
    delta = pd.Timedelta(minutes=int(tol/2))
    future_time = pd.to_datetime(current_time) + delta
    future_mask = (df['Time'] > current_time) & (df['Time'] <= future_time)
    future_len = len(df[future_mask])
    
    past_time = pd.to_datetime(current_time) - delta
    past_mask = (df['Time'] >= past_time) & (df['Time'] < current_time)
    past_len = len(df[past_mask])
    
    # If the future and past are same length
    if future_len == past_len:
        mask = (df['Time'] >= past_time) & (df['Time'] <= future_time)
        return df[mask]
    
    # Not enough data in the future
    elif future_len < past_len:
        delta = pd.Timedelta(minutes=tol - future_len)
        past_time = pd.to_datetime(current_time) - delta
        mask = (df['Time'] >= past_time) & (df['Time'] <= future_time)
        return df[mask]
    
    # Not enough data in the past
    elif future_len > past_len:
        delta = pd.Timedelta(minutes=tol - past_len)
        future_time = pd.to_datetime(current_time) + delta
        mask = (df['Time'] >= past_time) & (df['Time'] <= future_time)
        return df[mask]
    
    else:
        return None
    
def add_stability(df, tol, stable=STABLE, method='hybrid'):
    """
    Processes the DataFrame to add a 'Status' column with 'Stable' or 'Unstable' 
    labels based on the stability of the variables. Also tracks the number of
    unstable instances for each variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input Dataframe that contains the variables to be analyzed.
        
    tol : int
        Number of minutes to look into the future and past.
        
    Returns
    ----------
    df : pd.DataFrame
        The processed DataFrame with the 'Status' column and assigned labels.
    unstable_counter : dict
        Contains the number of unstable instances for each variable and how 
        many times they were unstable.
    """
    df["Status"] = None
    df['Time'] = pd.to_datetime(df['Time'])
    
    unstable_counter, counter = {}, [0]*STABILITY_LEN
    start_time, end_time = df['Time'].iloc[0], df['Time'].iloc[-1]
    mask = (df["Time"] >= start_time) & (df["Time"] <= end_time)
    
    for index, row in df[mask].iterrows():
        current_time = row['Time']
        
        if method == 'hybrid':
            df_period = time_frame_future_past(df, current_time, tol)
        elif method == 'future':
            df_period = time_frame_future(df, current_time, tol)
        
        if df_period is None:
            continue  

        df_stable, _ = stability(df, df_period["Time"].iloc[0], 
                                 df_period["Time"].iloc[-1],
                                 stable=stable)
        status_counts = df_stable['Status'].value_counts()
        
        if 'Stable' in status_counts:
            stable_count = status_counts['Stable'] 
            if stable_count == STABILITY_LEN:
                df.at[index, 'Status'] = "Stable"
            else:
                df.at[index, 'Status'] = "Unstable"
        else:
            stable_count = 0
            
        counter[stable_count-1] += 1

        for index, row in df_stable.iterrows():
            status = row['Status']
            
            if status != 'Stable':
                unstable_counter[index] = unstable_counter.get(index, 0) + 1

    unstable_counter["Counter"] = counter
    return df, unstable_counter

