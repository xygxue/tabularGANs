
def pad_leading_zero(df, col):
    for c in col:
        df[c] = df[c].apply(str).apply(lambda x: x.zfill(15))
    return df