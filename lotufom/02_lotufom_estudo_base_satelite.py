#%%
# Imports
import pandas as pd
import cv2
import numpy as np
from urllib import request

satelite = pd.read_parquet(r'..\dados\SATELITE_train.parquet')

# %%

url = satelite.path[0]

# Abre a URL e lê a imagem
resp = request.urlopen(url)

img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)

#%%
imagem = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

x_inicial, y_inicial = 50, 120  # Ponto inicial (topo esquerdo) do corte
x_final, y_final = 2100, 2200    # Ponto final (parte inferior direita) do corte

# Cortar a imagem
imagem_cortada = imagem[y_inicial:y_final, x_inicial:x_final]

import os
os.remove(r'C:\Users\mathe\OneDrive\Área de Trabalho\SCRIO\USINA\USINA_ITA\lotufom\img.jpg')

cv2.imwrite(r'img.jpg', imagem_cortada)
# %%
