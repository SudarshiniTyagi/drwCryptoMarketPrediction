import pandas as pd

train_data = pd.read_parquet('data/train.parquet')
# test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
print(train_data.head())