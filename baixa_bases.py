# %%
# Imports

import requests
import pandas as pd
from datetime import timedelta, datetime
from tqdm import tqdm
from glob import glob

# %%
def get_data_as_pandas(name:str):
    
    name = name.lower()
    assert name in ('bimtra', 'cat-62', 'esperas', 'metaf', 'metar', 'satelite', 'tc-prev', 'tc-real')
    
    #URL da API que você deseja acessar
    url = f'http://montreal.icea.decea.mil.br:5002/api/v1/{name}'

    #Seu token de autenticação
    token = 'a779d04f85c4bf6cfa586d30aaec57c44e9b7173'

    if name == 'cat-62':
        seconds = ' 00:00:00.000'
    else: 
        seconds = ''

    #Realize a requisição GET
    response = requests.get(
        url=url,
        params={
            'token': token, 
            'idate': '2022-06-01' + seconds, 
            'fdate': '2023-01-01' + seconds
            })
    
    assert response.status_code == 200

    res = response.json()

    return pd.DataFrame(res)


def get_data_cat_62(idate='2022-06-01', fdate='2023-01-01', period='30T'):
    
    #URL da API que você deseja acessar
    url = f'http://montreal.icea.decea.mil.br:5002/api/v1/cat-62'

    #Seu token de autenticação
    token = 'a779d04f85c4bf6cfa586d30aaec57c44e9b7173'

    idate = datetime.strptime(idate, '%Y-%m-%d')
    fdate = datetime.strptime(fdate, '%Y-%m-%d')
    days = (fdate - idate).days

    for i in tqdm(range(days + 1)):
        new_idate = idate + timedelta(i)
        new_fdate = idate + timedelta(i+1)

        #Realize a requisição GET
        response = requests.get(
            url=url,
            params={
                'token': token, 
                'idate': new_idate.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], 
                'fdate': new_fdate.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                })
        
        assert response.status_code == 200
        res = response.json()

        data = pd.DataFrame(res)

        if len(data) == 0:
            print(f"No data for {new_idate.strftime('%Y-%m-%d')} | {new_fdate.strftime('%Y-%m-%d')}")
            continue
        
        # Ajuste filtro de periodo
        data.dt_radar = pd.to_datetime(data.dt_radar/1000, unit='s')
        data = data.groupby('flightid').resample(period, on='dt_radar').mean(numeric_only=True).reset_index()

        if i == 0:
            data.to_csv(rf'dados/CAT62_train_{period}.csv', index=False)
        else:
            data.to_csv(rf'dados/CAT62_train_{period}.csv', mode='a', index=False, header=False)

    final_data = pd.read_csv(rf'dados/CAT62_train_{period}.csv')
    final_data.to_parquet(rf'dados/CAT62_train_{period}.parquet', index=False)
    
    return final_data

# %%
# Dados de Decolagem e Pouso
BIMTRA_train = get_data_as_pandas('bimtra')
BIMTRA_train.to_parquet(r'dados\BIMTRA_train.parquet', index=False)
BIMTRA_train.sample(2)

# %%
# Dados de Síntese Radar
period='10T'
if len(glob(r'dados\CAT62_train_{period}')) == 0:
    CAT62_train = get_data_cat_62(period=period)
else:
    CAT62_train = pd.read_parquet(rf'dados\CAT62_train_{period}.parquet')
CAT62_train.sample(2)
# %%
# Dados de Quantidades de Esperas em voo por hora
ESPERAS_train = get_data_as_pandas('esperas')
ESPERAS_train.to_parquet(r'dados\ESPERAS_train.parquet', index=False)
ESPERAS_train.sample(2)

# %%
# Previsão de Dados Meteorológicos em Aeródromos
METAF_train = get_data_as_pandas('metaf')
METAF_train.to_parquet(r'dados\METAF_train.parquet', index=False)
METAF_train.sample(2)

# %%
# Relatório de Dados Meteorológicos em Aeródromos
METAR_train = get_data_as_pandas('metar')
METAR_train.to_parquet(r'dados\METAR_train.parquet', index=False)
METAR_train.sample(2)

# %%
# Satélite Meteorológico
SATELITE_train = get_data_as_pandas('satelite')
SATELITE_train.to_parquet(r'dados\SATELITE_train.parquet', index=False)
SATELITE_train.sample(2)

# %%
# Previsão de Troca de Cabeceiras de Aeródromo
TC_PREV_train = get_data_as_pandas('tc-prev')
TC_PREV_train.to_parquet(r'dados\TC_PREV_train.parquet', index=False)
TC_PREV_train.sample(2)

# %%
# Histórico da Troca de Cabeceiras de Aeródromo
TC_REAL_train = get_data_as_pandas('tc-real')
TC_REAL_train.to_parquet(r'dados\TC_REAL_train.parquet', index=False)
TC_REAL_train.sample(2)



# %%
