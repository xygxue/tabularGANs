import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from CTAB_GAN.model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
from cswgan.lib.plot import compare_hists

cur_date = '2022-04-13'
exp = 2
real_datafile = 'labelencode_trans_3.csv'
fake_datafile = 'labelencode_tablegan_fake_3_2022_04_13.csv'
CAT_COL = ['account_id', 'type', 'operation', 'k_symbol', 'bank', 'account']
CLASSIFIER = ["lr", "dt", "rf", "mlp", "svm"]
DATASET = "czech_bank"


REAL_PATH = os.path.join(Path(__file__).parent, 'resources/real_datasets', DATASET, real_datafile)
FAKE_PTAHS = [os.path.join(Path(__file__).parent, 'resources/fake_datasets', DATASET, fake_datafile)]
RESULT_PATH = os.path.join(Path(__file__).parent, 'resources/eval_results')
PLOT_PATH = os.path.join(Path(__file__).parent, 'resources/plot', DATASET)


def ml_utility_eval(real_path, fake_paths, classifiers_list, result_path):
    '''ML Utility Evaluation, compare the scores between classifiers that are Train on Real Test on Real and
    Train on Fake and Test on Real'''
    result_mat = get_utility_metrics(real_path, fake_paths, "MinMax", classifiers_list, test_ratio=0.20)
    result_df_ml = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
    result_df_ml.index = classifiers_list

    result_df_ml.to_csv(os.path.join(result_path, 'ml_utility', fake_datafile))


def stat_eval(real_path, fake_paths, result_path, col_categorical):
    # Statistical Similarity Evaluation
    stat_res_avg = []
    for f_pth in fake_paths:
        stat_res = stat_sim(real_path, f_pth, col_categorical)
        stat_res_avg.append(stat_res)

    stat_columns = ["Average WD (Continuous Columns)", "Average JSD (Categorical Columns)", "Correlation Distance"]
    stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns)

    stat_results.to_csv(os.path.join(result_path, 'stat',fake_datafile))


def privacy_eval(real_path, fake_paths, result_path):
    # Nearest Neighbour Privacy Analysis
    priv_res_avg = []
    for fake_path in fake_paths:
        priv_res = privacy_metrics(real_path, fake_path)
        priv_res_avg.append(priv_res)

    privacy_columns = ["DCR between Real and Fake (5th perc)", "DCR within Real(5th perc)", "DCR within Fake (5th perc)",
                       "NNDR between Real and Fake (5th perc)", "NNDR within Real (5th perc)",
                       "NNDR within Fake (5th perc)"]
    privacy_results = pd.DataFrame(np.array(priv_res_avg).mean(axis=0).reshape(1, 6), columns=privacy_columns)

    privacy_results.to_csv(os.path.join(result_path, 'privacy',fake_datafile))


def plot_dist(real_path, fake_paths, plot_path):
    # plot the histogram for each column in fake and real data
    for f_pth in fake_paths:
        x_fake = pd.read_csv(f_pth)
        x_real = pd.read_csv(real_path)
        directory = os.path.splitext(fake_datafile)[0]
        os.mkdir(os.path.join(plot_path, directory))
        for c in x_fake.columns:
            fake_col = x_fake[c].to_numpy()
            real_col = x_real[c].to_numpy()
            ax = compare_hists(real_col, fake_col, ax=None, log=False, label=None)
            ax.plot()
            plt.savefig(os.path.join(plot_path, directory, f"{c}.png"))


if __name__ == '__main__':
    ml_utility_eval(REAL_PATH, FAKE_PTAHS, CLASSIFIER, RESULT_PATH)
    stat_eval(REAL_PATH, FAKE_PTAHS, RESULT_PATH, CAT_COL)
    privacy_eval(REAL_PATH, FAKE_PTAHS, RESULT_PATH)
