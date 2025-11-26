"""
Embeddings.py
-------------

Este script permite generar embeddings de jugadores de la NBA utilizando el encoder de un autoencoder previamente entrenado.
Procesa los datos de los jugadores, obtiene su representación latente (embedding) y guarda el resultado en un archivo CSV.
Incluye funciones para cargar los embeddings desde CSV y devolverlos en formato diccionario. 
"""

import torch
import pandas as pd
import numpy as np
from MLP import MLP
from Autoencoder import Autoencoder
from Guardar_Cargar import cargar_modelo
from Procesar_datos_AE import procesar_datos


def generar_embeddings( autoencoder_path,
                        players_path,
                        modo_columnas: str = "completo"
                        ):
    """
    Genera y devuelve un diccionario {player_id: embedding_numpy_array} usando el encoder
    del autoencoder cargado en `autoencoder_path` aplicado a los datos en `players_path`.
    No guarda en disco (responsabilidad de `guardar_embeddings`).
    """

    # ---------- Cargar modelo y procesar datos ----------
    autoencoder = cargar_modelo(autoencoder_path)
    datos_jugadores, etiquetas = procesar_datos(players_path, modo_columnas)

    # ---------- GENERAR EMBEDDINGS ----------
    with torch.no_grad():
        embeddings_tensor = autoencoder.encoder(datos_jugadores)  # shape (num_jugadores, latent_size)
        embeddings_np = embeddings_tensor.detach().cpu().numpy()  # convertimos todo a NumPy

    # Creamos dicc jugador: embedding (aseguramos tipo float32)
    player_ids = etiquetas["Player"].values
    embeddings_dict = {jugador: emb for jugador, emb in zip(player_ids, embeddings_np)}
    
    embeddings_dict = {jugador: np.asarray(emb, dtype=np.float32) for jugador, emb in zip(player_ids, embeddings_np)}
    
    return embeddings_dict

def guardar_embeddings(embeddings_dict, embeddings_path):
    """
    Guarda `embeddings_dict` en `embeddings_path` en CSV.

    Parámetros:
    - embeddings_dict: dict de la forma {clave: np_array_like}
    - embeddings_path: ruta al archivo CSV destino
    
    """

    if not isinstance(embeddings_dict, dict):
        raise TypeError("embeddings_dict debe ser un dict de clave->array")

    # Transfomar valores a numpy
    claves = list(embeddings_dict.keys())
    if len(claves) == 0:
        # crear csv vacío
        pd.DataFrame().to_csv(embeddings_path, index=False)
        return

    values = [np.asarray(emb) for emb in (embeddings_dict[k] for k in claves)]

    # comprobar que todas las embeddings tienen la misma dimensión
    dim0 = values[0].shape[0]
    if not all(v.shape[0] == dim0 for v in values):
        raise ValueError("Todas las embeddings deben tener la misma longitud")

    # fila por jugador: Player, dim_0, dim_1, ...
    embeddings_list = [[k] + v.tolist() for k, v in zip(claves, values)]
    column_names = ['Player'] + [f"dim_{i}" for i in range(dim0)]
    df_embeddings = pd.DataFrame(embeddings_list, columns=column_names)
    df_embeddings.to_csv(embeddings_path, index=False)


def cargar_embeddings(csv_file):
    df = pd.read_csv(csv_file)
    embeddings_dict = {}
    for _, row in df.iterrows():
        jugador = row['Player']
        embedding = row.iloc[1:].to_numpy(dtype=np.float32)
        embeddings_dict[jugador] = embedding
    return embeddings_dict

