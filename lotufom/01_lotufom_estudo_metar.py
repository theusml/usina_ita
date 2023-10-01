#%%
# Imports
import pandas as pd
import re
import numpy as np

#%%
# Import dos dados

dados = pd.read_parquet(r'../dados/dados_dummy_model.parquet')

#%%
# Isolando os piores scores para analise aprofundada

dados_piores_scores = dados.sort_values(
'tempo_diff', ascending=False).head(100)
dados_piores_scores.to_parquet(r'../dados/piores_scores_modelo_dummy.parquet', index=False)

#%%
# Separando os 30% piores

ax = dados_piores_scores.hist('tempo_diff')
estudo = dados_piores_scores[dados_piores_scores.tempo_diff <= 0.3]

#%%
# Hipóteses
# H0: Não há influencia significativa do clima nos casos.
# HA: O clima possui influencia nos casos do estudo, e comprometeram a previsão.
# Significancia de 0.05

clima =  pd.read_parquet(r'..\dados\METAR_train.parquet')

#%%
# Função para extrair informações do metar

def dummify_metar(metar_code: str):
# Expressões regulares para cada informação relevante

    angle_var, wind_cte, wind_gust, visibility, clouds, precipitation = [None]*6

    wind_cte = re.search(r'(\d{5}KT)|(\d{2}G)', metar_code)

    angle_var = re.search(r'(\d{3}V\d{3})', metar_code)

    wind_gust = re.search(r'(G\d{2}KT)', metar_code)

    visibility = re.search(r'(\s\d{4}\s)', metar_code)

    clouds = re.search(r'(FEW)|(SCT)|(BKN)|(OVC)|(CAVOK)|(SKC)|(NSC)|(NCD)', metar_code)

    precipitation = re.search(r'([\+\-\s]MI\s)|([\+\-\s]BC\s)|([\+\-\s]PR\s)|([\+\-\s]DR\s)|([\+\-\s]BL\s)|([\+\-\s]SH\s)|([\+\-\s]FZ\s)|([\+\-\s]BR\s)|([\+\-\s]FG\s)|([\+\-\s]FU\s)|([\+\-\s]VA\s)|([\+\-\s]DU\s)|([\+\-\s]SA\s)|([\+\-\s]HZ\s)|([\+\-\s]DZ\s)|([\+\-\s]RA\s)|([\+\-\s]SN\s)|([\+\-\s]SG\s)|([\+\-\s]PL\s)|([\+\-\s]GR\s)|([\+\-\s]GS\s)|([\+\-\s]PO\s)|([\+\-\s]SQ\s)|([\+\-\s]FC\s)|([\+\-\s]SS\s)|([\+\-\s]DS\s)|([\+\-\s]TS\s)|([\+\-\s]TS\s)', metar_code)

    if wind_cte:
        wind_cte = int(wind_cte[0][-4:].strip('KT').strip('G'))

    if angle_var:
        angle_var = angle_var[0].split('V')
        angle_var = int(angle_var[1]) - int(angle_var[0])

    if wind_gust:
        wind_gust = int(wind_gust[0].strip('KT').strip('G'))

    if visibility:
        visibility = int(visibility[0])

    if clouds:
        clouds = clouds[0]

    if precipitation:
        precipitation = precipitation[0][1:-1]

    if (visibility==None) & (clouds in ('CAVOK', 'SKC', 'NSC', 'NCD')):
        visibility=9999


    return (angle_var, wind_cte, wind_gust, visibility, clouds, precipitation)

#%%
# Decodificando o metar

clima['metar_dummy'] = clima.metar.apply(dummify_metar)

new_metar = clima['metar_dummy'].apply(pd.Series)
new_metar.columns = ['angle_var', 'wind_cte', 'wind_gust', 'visibility', 'clouds', 'precipitation']

clima = clima.merge(new_metar, how='left', left_index=True, right_index=True, validate='1:1')

#%%
# Ajustando as variaveis

clima.clouds = clima.clouds.map({
    np.nan:0,
    'CAVOK':0,
    'SKC':0,
    'NSC':0,
    'NCD':0,
    'FEW':1,
    'SCT':2,
    'BKN':3,
    'OVC':4})

clima.precipitation = clima.precipitation.map(
{   None: 0, np.nan:0,
    'MI':1,'BC':1,'PR':1,'DR':1,'BL':1,'SH':1,'FZ':1,'BR':2,'FG':2,'FU':2,'VA':2,'DU':2,'SA':2,'HZ':2,'DZ':3,'RA':3,'SN':3,'SG':3,'PL':3,'GR':3,'GS':3,'PO':4,'SQ':4,'FC':4,'SS':4,'DS':4,'TS':4,'TS':4
})

clima.wind_gust = clima.wind_gust.fillna(0)
clima.clouds = clima.clouds.fillna(0)
clima.angle_var = clima.angle_var.fillna(0)
clima.wind_gust = clima.wind_gust.fillna(0)

#%%
# Ajustando para 10T

clima.hora = pd.to_datetime(clima.hora/1000, unit='s')

#%%
clima_new = clima.groupby('aero').resample('60T', on='hora').mean(numeric_only=True).reset_index()

clima_new = clima_new.dropna()

colunas_float = clima_new.select_dtypes(include='float64')
clima_new[colunas_float.columns] = colunas_float.applymap(int)

# %%
