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
dados = pd.read_parquet(r'..\dados\BIMTRA_train.parquet')

#%%
# Removendo outliers
dados = dados[dados.origem != dados.destino]
dados = dados.reset_index(drop=True)

#%%
# Criando a feature de Y

dados['tempo'] = dados.dt_arr - dados.dt_dep

scaler = MinMaxScaler()
scaler = MinMaxScaler()

dados['tempo'] = scaler.fit_transform(dados[['tempo']].values)

# %%
# Scorando o modelo

splits = 50
kf = KFold(splits, shuffle=True, random_state=200)

results = []
for train_index, test_index in tqdm(kf.split(dados), total=splits):

    data_train, data_test = dados.loc[train_index], dados.loc[test_index]
    y_true = data_test.tempo

    modelo = smf.ols(formula='tempo ~ C(origem) : C(destino)', data=data_train).fit()

    y_pred = modelo.predict(data_test)

    results.append(r2_score(y_true, y_pred))

u = np.mean(results) 
alpha = np.std(results)
print(f'Model Score: {round(u, 4)} ± {round(1.96*alpha, 3)}')

# %%
# Final dummy model
dummy_model = smf.ols(formula='tempo ~ C(origem) : C(destino)', data=dados).fit()
y_pred = dummy_model.predict(dados)

#%%
# Capturando os maiores ofensores
dados['tempo_pred'] = y_pred
dados['tempo_diff'] = abs(dados.tempo_pred - dados.tempo)

dados.to_parquet(r'dados_dummy_model.parquet', index=False)

dados_piores_scores = dados.sort_values(
'tempo_diff', ascending=False).head(20)
dados_piores_scores.to_parquet('piores_scores_modelo_dummy.parquet', index=False)
# %%
