import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

DATASET = 'czech_bank'
MAPFILE = 'mapping.pickle'
LE_MAP_PATH = os.path.join(Path(__file__).parent, 'resources/label_encode_map', DATASET, MAPFILE)
CAT_COL = ['account_id', 'type', 'operation', 'k_symbol', 'bank', 'account']
DATE_COLUMN = 'date'


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


def time_diff(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    diff = df.loc[:, [date_column]].sub(df.loc[0, [date_column]], axis='columns')/np.timedelta64(1, "D").astype("int64")
    df[date_column] = diff[date_column].dt.days
    return df


def preprocess(filename, date_column, le_map_path, dataset):
    csv_path = os.path.join(Path(__file__).parents[0], 'resources/real_datasets', dataset, filename)
    raw = pd.read_csv(csv_path)
    raw[['amount', 'balance']] = raw[['amount', 'balance']].astype(int)
    # cyclical_encode(raw, date_column)
    df = raw.drop(columns=['trans_id'])
    df[date_column]= pd.to_datetime(df[date_column])
    # label_encode(df, cat_col, le_map_path)
    # raw_one_hot = one_hot(raw, one_hot_col)
    return df


# processed = preprocess('clean_trans.csv', DATE_COLUMN, LE_MAP_PATH, DATASET)
#
# splited_df = np.array_split(processed, 20)
# splited_df[0].to_csv(os.path.join(Path(__file__).parents[0], 'resources/real_datasets', dataset, 'trans_3.csv'), index=False)


def process(filename, cat_col, date_column, dataset):
    csv_path = os.path.join(Path(__file__).parents[0], 'resources/real_datasets', dataset, filename)
    df = pd.read_csv(csv_path)
    df['first_day'] = '1993-01-01'
    df['first_day'] = pd.to_datetime(df['first_day'])
    df[date_column] = pd.to_datetime(df[date_column])
    df[date_column] -= df['first_day']
    df[date_column] = df[date_column].dt.days
    df[cat_col] = df[cat_col].apply(LabelEncoder().fit_transform)
    df_out = df.drop(columns=['first_day'])
    df_out.to_csv(os.path.join(Path(__file__).parents[0], 'resources/real_datasets', dataset, 'labelencode_' + filename),
              index=False)


def min_acc():
    csv_path = os.path.join(Path(__file__).parents[0], 'resources/real_datasets', 'czech_bank', 'clean_trans.csv')
    df = pd.read_csv(csv_path)
    df = df.set_index('date')
    accs = df.account_id.unique()
    min_acc = 100000
    grp = df.groupby(['account_id'])
    min_len = grp.size().min()
    return min_len


def tablegan_data(filename, dataset):
    csv_path = os.path.join(Path(__file__).parents[0], 'resources/real_datasets', dataset, f"{filename}.csv")
    raw = pd.read_csv(csv_path)

    def ctpy_check(row):
        if row['bank'] == 13 and row['account'] == 1715:
            return 0
        else:
            return 1
    label_df = pd.DataFrame()
    label_df['label'] = raw.apply(lambda row: ctpy_check(row), axis=1)
    label_df.to_csv(os.path.join(Path(__file__).parents[0],
                                 'resources/real_datasets',
                                 dataset, f"{filename}_labels.csv"),
                    index=False, header=False)


tablegan_data('labelencode_trans_3', DATASET)
