#%%
# Imports

import requests
import pandas as pd

#%%
BIMTRA_train = pd.read_parquet(r'..\dados\BIMTRA_train.parquet')
# %%
# Tive que divir por 1000, pois tinhamos 3 zeros a mais no final do timestamp. Provavel para datas futuras.

BIMTRA_train.dt_dep = pd.to_datetime(BIMTRA_train.dt_dep/1000, unit='s')
BIMTRA_train.dt_arr = pd.to_datetime(BIMTRA_train.dt_arr/1000, unit='s')
BIMTRA_train.head()

# %%
BIMTRA_train[BIMTRA_train.dt_dep== BIMTRA_train.dt_dep.min()]

# %%
