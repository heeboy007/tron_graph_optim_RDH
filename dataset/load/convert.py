import numpy as np
import pandas as pd

def convert_dataframe(df: pd.DataFrame):
    src = df["src"].to_numpy(dtype=np.int32)
    dst = df["dst"].to_numpy(dtype=np.int32)
    bn = df['bn'].to_numpy(dtype=np.int32)
    is_d = df['is_d'].to_numpy(dtype=np.bool)
    return (src, dst, bn, is_d)