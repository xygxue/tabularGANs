import os
from pathlib import Path
import numpy as np
import pandas as pd


def one_hot(df, column):
    return pd.get_dummies(df[column])


def cyclical_encode(df, date_column):
    date_col = pd.DatetimeIndex(df[date_column])
    df[date_column] = date_col
    df['year'] = date_col.year
    df['month'] = date_col.month
    df['day'] = date_col.day
    df['dayofweek'] = date_col.dayofweek


def preprocess(filename, date_column):
    csv_path = os.path.join(Path(__file__).parents[0], 'resources/', filename)
    raw = pd.read_csv(csv_path)
    raw[['amount', 'balance']] = raw[['amount', 'balance']].astype(int)
    cyclical_encode(raw, date_column)
    # raw_one_hot = one_hot(raw, one_hot_col)
    df = raw.drop(columns=['date', 'trans_id'])

    return df


processed = preprocess('clean_trans.csv', 'date')

splited_df = np.array_split(processed, 20)
splited_df[0].to_csv(os.path.join(Path(__file__).parents[0], 'resources/', 'trans_0.csv'), index=False)

