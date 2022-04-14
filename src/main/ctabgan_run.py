import os
from datetime import date
from pathlib import Path


from CTAB_GAN.model.ctabgan import CTABGAN

# Specifying the replication number
num_exp = 1

dataset = "czech_bank"
real_datafile = 'labelencode_trans_3.csv'
real_path = os.path.join(Path(__file__).parent, 'resources/real_datasets', dataset, real_datafile)
fake_path = os.path.join(Path(__file__).parent, 'resources/fake_datasets', dataset, 'ctabgan_fake_{exp}_{cur_date}.csv')


synthesizer = CTABGAN(raw_csv_path=real_path,
                      test_ratio=0.20,
                      categorical_columns=['account_id', 'type', 'operation', 'k_symbol', 'bank', 'account'],
                      log_columns=['amount', 'balance'],
                      mixed_columns={'k_symbol': [0.0], 'bank': [0.0], 'account': [0.0]},
                      integer_columns=['amount', 'balance'],
                      problem_type={"Classification": 'operation'},
                      epochs=100)


if __name__ == '__main__':
    # Fitting the synthesizer to the training dataset and generating synthetic data
    today = date.today()
    cur_date = today.strftime("%Y-%m-%d")

    for i in range(num_exp):
        synthesizer.fit()
        syn = synthesizer.generate_samples()
        syn.to_csv(fake_path.format(exp=i, cur_date=cur_date), index=False)
