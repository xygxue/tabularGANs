import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from CTAB_GAN.model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics

cur_date = '2022-03-29'
exp = 0
classifiers_list = ["lr", "dt", "rf", "mlp", "svm"]
dataset = "czech_bank"
real_datafile = 'trans_1.csv'
fake_datafile = 'fake_{exp}_{cur_date}.csv'.format(exp=exp, cur_date=cur_date)

result_file = 'result_' + fake_datafile
result_dict = dict()

real_path = os.path.join(Path(__file__).parent, 'resources/real_datasets', dataset, real_datafile)
fake_paths = [os.path.join(Path(__file__).parent, 'resources/fake_datasets', dataset, fake_datafile)]
result_path = os.path.join(Path(__file__).parent, 'resources/eval_results/', result_file)

# ML Utility Evaluation
result_mat = get_utility_metrics(real_path, fake_paths, "MinMax", classifiers_list, test_ratio=0.30)
result_df_ml = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
result_df_ml.index = classifiers_list

result_dict['ml_utility'] = result_df_ml

col_categorical = ['account_id', 'type', 'operation', 'k_symbol', 'bank', 'account']

# Statistical Similarity Evaluation
stat_res_avg = []
for fake_path in fake_paths:
    stat_res = stat_sim(real_path, fake_path, col_categorical)
    stat_res_avg.append(stat_res)

stat_columns = ["Average WD (Continuous Columns)", "Average JSD (Categorical Columns)", "Correlation Distance"]
stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns)
result_dict['stat'] = stat_results

# Nearest Neighbour Privacy Analysis
priv_res_avg = []
for fake_path in fake_paths:
    priv_res = privacy_metrics(real_path, fake_path)
    priv_res_avg.append(priv_res)

privacy_columns = ["DCR between Real and Fake (5th perc)", "DCR within Real(5th perc)", "DCR within Fake (5th perc)",
                   "NNDR between Real and Fake (5th perc)", "NNDR within Real (5th perc)",
                   "NNDR within Fake (5th perc)"]
privacy_results = pd.DataFrame(np.array(priv_res_avg).mean(axis=0).reshape(1, 6), columns=privacy_columns)
result_dict['privacy'] = privacy_results


with open(result_path, 'w') as fp:
    json.dump(result_dict, fp)
