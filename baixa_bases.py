# %%
# Imports

import requests
import pandas as pd
import json

# %%
def get_data_as_pandas(name:str):
    
    name = name.lower()
    assert name in ('bimtra', 'cat-62', 'esperas', 'metaf', 'metar', 'satelite', 'tc-prev', 'tc-real')
    
    #URL da API que você deseja acessar
    url = f'http://montreal.icea.decea.mil.br:5002/api/v1/{name}'

    #Seu token de autenticação
    token = 'a779d04f85c4bf6cfa586d30aaec57c44e9b7173'

    #Realize a requisição GET
    response = requests.get(url=url,params={'token':token, 'idate':'2022-06-01', 'fdate':'2023-01-01'})
    assert response.status_code == 200

    res = response.json()

    return pd.DataFrame(res)

# BIMTRA_train

# %%
# Principal
BIMTRA_train = get_data_as_pandas('bimtra')
BIMTRA_train.to_parquet(r'dados\BIMTRA_train.parquet', index=False)
BIMTRA_train.sample(2)

# %%
CAT62_train = get_data_as_pandas('cat-62')
CAT62_train.to_parquet(r'dados\CAT62_train.parquet', index=False)
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
# Previsão de Troca de Cabeceiras de Pista
TC_PREV_train = get_data_as_pandas('tc-prev')
TC_PREV_train.to_parquet(r'dados\TC_PREV_train.parquet', index=False)
TC_PREV_train.sample(2)

# %%
# Previsão de Troca de Cabeceiras de Pista
TC_REAL_train = get_data_as_pandas('tc-real')
TC_REAL_train.to_parquet(r'dados\TC_REAL_train.parquet', index=False)
TC_REAL_train.sample(2)




