import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing

dataset = 'czech_bank'
mapfile = 'mapping.pickle'
le_map_path = os.path.join(Path(__file__).parent, 'resources/label_encode_map', dataset, mapfile)
cat_col = ['account_id', 'type', 'operation', 'k_symbol', 'bank', 'account']


def label_encode(df, cat_col, mapping_path):
    le_name_mapping = dict()
    le = preprocessing.LabelEncoder()
    for col in cat_col:
        df[col] = le.fit_transform(df[col])
        le_name_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    with open(mapping_path, 'wb') as fp:
        pickle.dump(le_name_mapping, fp)


def one_hot(df, column):
    return pd.get_dummies(df[column])


def cyclical_encode(df, date_column):
    date_col = pd.DatetimeIndex(df[date_column])
    df[date_column] = date_col
    df['year'] = date_col.year
    df['month'] = date_col.month
    df['day'] = date_col.day
    df['dayofweek'] = date_col.dayofweek


def preprocess(filename, date_column, le_map_path):
    csv_path = os.path.join(Path(__file__).parents[0], 'resources/real_datasets', dataset, filename)
    raw = pd.read_csv(csv_path)
    raw[['amount', 'balance']] = raw[['amount', 'balance']].astype(int)
    # cyclical_encode(raw, date_column)
    df = raw.drop(columns=['trans_id'])
    df[date_column]= pd.to_datetime(df[date_column])
    # label_encode(df, cat_col, le_map_path)
    # raw_one_hot = one_hot(raw, one_hot_col)
    return df


processed = preprocess('clean_trans.csv', 'date', le_map_path)

splited_df = np.array_split(processed, 5)
splited_df[0].to_csv(os.path.join(Path(__file__).parents[0], 'resources/real_datasets', dataset, 'trans_4.csv'), index=False)

