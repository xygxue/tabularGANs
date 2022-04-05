import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from CTAB_GAN.model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
from cswgan.lib.plot import compare_hists

cur_date = '2022-03-31'
exp = 2
real_datafile = 'labelencode_trans_3.csv'
fake_datafile = 'labelencode_ctgan_fake_3_2022-03-31.csv'.format(exp=exp, cur_date=cur_date)

classifiers_list = ["lr", "dt", "rf", "mlp", "svm"]
dataset = "czech_bank"

real_path = os.path.join(Path(__file__).parent, 'resources/real_datasets', dataset, real_datafile)
fake_paths = [os.path.join(Path(__file__).parent, 'resources/fake_datasets', dataset, fake_datafile)]
result_path = os.path.join(Path(__file__).parent, 'resources/eval_results')

# ML Utility Evaluation
# result_mat = get_utility_metrics(real_path, fake_paths, "MinMax", classifiers_list, test_ratio=0.30)
# result_df_ml = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
# result_df_ml.index = classifiers_list
#
# result_file_ml = 'ml_utility_' + fake_datafile
# result_df_ml.to_csv(os.path.join(result_path, result_file_ml))

col_categorical = ['account_id', 'type', 'operation', 'k_symbol', 'bank', 'account']

# Statistical Similarity Evaluation
# stat_res_avg = []
# for fake_path in fake_paths:
#     stat_res = stat_sim(real_path, fake_path, col_categorical)
#     stat_res_avg.append(stat_res)
#
# stat_columns = ["Average WD (Continuous Columns)", "Average JSD (Categorical Columns)", "Correlation Distance"]
# stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns)
#
# result_file_stat = 'stat_' + fake_datafile
# stat_results.to_csv(os.path.join(result_path, result_file_stat))

# Nearest Neighbour Privacy Analysis
priv_res_avg = []
for fake_path in fake_paths:
    priv_res = privacy_metrics(real_path, fake_path)
    priv_res_avg.append(priv_res)

privacy_columns = ["DCR between Real and Fake (5th perc)", "DCR within Real(5th perc)", "DCR within Fake (5th perc)",
                   "NNDR between Real and Fake (5th perc)", "NNDR within Real (5th perc)",
                   "NNDR within Fake (5th perc)"]
privacy_results = pd.DataFrame(np.array(priv_res_avg).mean(axis=0).reshape(1, 6), columns=privacy_columns)
result_file_pri = 'privacy_' + fake_datafile
privacy_results.to_csv(os.path.join(result_path, result_file_pri))


# plot the histogram for each column in fake and real data
x_fake = pd.read_csv(fake_paths[0])
x_real = pd.read_csv(real_path)
compare_hists(x_real, x_fake, ax=None, log=False, label=None)
