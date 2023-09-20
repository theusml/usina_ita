# Arquivo com o modelo dummy para a base de score

#%%
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from tqdm import tqdm

#%%
# Leitura dos dados
dados = pd.read_excel(r'..\dados\BIMTRA_train.xlsx')

#%%
# Removendo outliers
dados = dados[dados.origem != dados.destino]
dados = dados.reset_index(drop=True)

#%%
# Criando a feature de Y

dados['tempo'] = dados.dt_arr - dados.dt_dep

scalar = MinMaxScaler()
scalar = MinMaxScaler()

dados['tempo'] = scalar.fit_transform(dados[['tempo']].values)

# %%
# Scorando o modelo

splits = 25
kf = KFold(splits, shuffle=True, random_state=200)

results = []
for train_index, test_index in tqdm(kf.split(dados), total=splits):

    data_train, data_test = dados.loc[train_index], dados.loc[test_index]
    y_true = data_test.tempo

    modelo = smf.ols(formula='tempo ~ C(origem) : C(destino)', data=data_train).fit()

    y_pred = modelo.predict(sm.add_constant(data_test))

    results.append(r2_score(y_true, y_pred))

u = np.mean(results) 
alpha = np.std(results)
print(f'Model Score: {round(u, 4)} ± {round(1.96*alpha, 3)}')

# %%

# %%

