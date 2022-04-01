import os
from datetime import date
from pathlib import Path

import pandas as pd
from sdv.tabular import CTGAN

exp = 4
model_name = 'ctgan'
dataset = "czech_bank"
real_datafile = 'trans_4.csv'
real_path = os.path.join(Path(__file__).parent, 'resources/real_datasets', dataset, real_datafile)
fake_path = os.path.join(Path(__file__).parent, 'resources/fake_datasets', dataset,
                         '{model_name}_fake_{exp}_{cur_date}.csv')
today = date.today()
cur_date = today.strftime("%Y-%m-%d")

real_data = pd.read_csv(real_path)
real_data_len = real_data.shape[0]
real_data['date'] = pd.to_datetime(real_data['date'])
model = CTGAN(rounding=0)

model.fit(real_data)

fake_data = model.sample(real_data_len)

fake_data.to_csv(fake_path.format(model_name=model_name, exp=exp, cur_date=cur_date), index=False)
