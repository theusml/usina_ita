#%%
# Imports
import pandas as pd
import re

#%%
# Import dos dados

dados = pd.read_parquet(r'../dados/dados_dummy_model.parquet')

#%%
# Isolando os piores scores para analise aprofundada

dados_piores_scores = dados.sort_values(
'tempo_diff', ascending=False).head(100)
dados_piores_scores.to_parquet(r'../dados/piores_scores_modelo_dummy.parquet', index=False)

#%%
# Separando os 50% piores

ax = dados_piores_scores.hist('tempo_diff')

estudo = dados_piores_scores[dados_piores_scores.tempo_diff <= 0.3]

#%%
# Hipóteses
# H0: Não há influencia significativa do clima nos casos.
# HA: O clima possui influencia nos casos do estudo, e comprometeram a previsão.
# Significancia de 0.05

clima =  pd.read_parquet(r'..\dados\METAR_train.parquet')

#%%

def dummify_metar(metar_code: str):
# Expressões regulares para cada informação relevante

    angle_var, wind_cte, wind_gust, visibility, clouds, precipitacao = [None]*6

    wind_cte = re.search(r'(\d{5}KT)|(\d{2}G)', metar_code)

    angle_var = re.findall(r'(\d{3}V\d{3})', metar_code)

    wind_gust = re.findall(r'(G\d{2}KT)', metar_code)

    visibility = re.findall(r'(\s\d{4}\s)', metar_code)

    clouds = re.search(r'(FEW)|(SCT)|(BKN)|(OVC)', metar_code)

    precipitacao = re.search(r'([\+\-\s]DZ )|([\+\-\s]RA )|([\+\-\s]PL )|([\+\-\s]GR )|([\+\-\s]GS )|([\+\-\s]BR )|([\+\-\s]FG )|([\+\-\s]FU )|([\+\-\s]DU )|([\+\-\s]HZ )|([\+\-\s]SQ )|([\+\-\s]FC )|([\+\-\s]TS )', metar_code)


    if wind_cte:
        wind_cte = int(wind_cte[0][-4:].strip('KT').strip('G'))

    if angle_var:
        angle_var = angle_var[0].split('V')
        angle_var = int(angle_var[1]) - int(angle_var[0])

    if wind_gust:
        wind_gust = wind_gust[0].strip('KT').strip('G')

    if visibility:
        visibility = visibility[0]

    if clouds:
        clouds = clouds[0]

    if precipitacao:
        precipitacao = precipitacao[0]

    return (angle_var, wind_cte, wind_gust, visibility, clouds, precipitacao)
#%%

clima['metar_dummy'] = clima.metar.apply(dummify_metar)

new_metar = clima['metar_dummy'].apply(pd.Series)
new_metar.columns = ['angle_var', 'wind_cte', 'wind_gust', 'visibility', 'clouds', 'precipitacao']

clima = clima.merge(new_metar, how='left', left_index=True, right_index=True, validate='1:1')
# %%
# Tive que divir por 1000, pois tinhamos 3 zeros a mais no final do timestamp. Provavel para datas futuras.

#BIMTRA_train.dt_dep = pd.to_datetime(BIMTRA_train.dt_dep/1000, unit='s')
#BIMTRA_train.dt_arr = pd.to_datetime(BIMTRA_train.dt_arr/1000, unit='s')
#BIMTRA_train.head()

# %%
#BIMTRA_train[BIMTRA_train.dt_dep== BIMTRA_train.dt_dep.min()]

# %%
