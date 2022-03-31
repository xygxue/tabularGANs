import os
from pathlib import Path

import pandas as pd
from sdv.tabular import CTGAN

model_name = 'ctgan'
dataset = "czech_bank"
real_datafile = 'trans_3.csv'
real_path = os.path.join(Path(__file__).parent, 'resources/real_datasets', dataset, real_datafile)
fake_path = os.path.join(Path(__file__).parent, 'resources/fake_datasets', dataset,
                         '{model_name}_fake_{exp}_{cur_date}.csv')

real_data = pd.read_csv(real_path)
real_data_len = real_data.shape[0]
real_data['date'] = pd.to_datetime(real_data['date'])
model = CTGAN(rounding=0)

model.fit(real_data)

fake_data = model.sample(real_data_len)
