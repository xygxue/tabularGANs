import numpy as np


def pad_leading_zero(df, col):
    for c in col:
        df[c] = df[c].apply(str).apply(lambda x: x.zfill(12) if x is not np.nan else x)
    return df
