import os
import numpy as np
import pandas as pd

def lysekil():
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

    # Check if the pickled file exists and load it
    if os.path.exists("../data/raw/cached_Lysekil.pkl"):
        raw_df = pd.read_pickle("../data/raw/cached_Lysekil.pkl")
    else:
        raw_df = pd.read_excel("../data/raw/Lysekil_MEA_data_unprotected.xlsx", 
                            header=None, 
                            names=headers, 
                            usecols=import_cols, 
                            skiprows=skip)
        raw_df.to_pickle("../data/raw/cached_Lysekil.pkl")
        
    return raw_df


