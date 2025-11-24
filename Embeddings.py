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
                        embeddings_path,
                        modo_columnas:str="completo"
                        ):
    
    # ---------- Cargar modelo y procesar datos ----------
    autoencoder = cargar_modelo(autoencoder_path)
    datos_jugadores,etiquetas = procesar_datos(players_path,modo_columnas)

    # ---------- GENERAR EMBEDDINGS ----------
    with torch.no_grad():
        embeddings_tensor = autoencoder.encoder(datos_jugadores)  # shape (num_jugadores, latent_size)
        embeddings_np = embeddings_tensor.numpy()  # convertimos todo a NumPy

    # Creamos dicc jugador: embedding
    player_ids = etiquetas["Player"].values
    embeddings_dict = {jugador: emb for jugador, emb in zip(player_ids, embeddings_np)}

    # ---------- GUARDAR EN CSV ----------
    embeddings_list = [[jugador] + emb.tolist() for jugador, emb in embeddings_dict.items()]
    column_names = ['Player'] + [f"dim_{i}" for i in range(embeddings_np.shape[1])]
    df_embeddings = pd.DataFrame(embeddings_list, columns=column_names)
    df_embeddings.to_csv(embeddings_path, index=False)

    print(f"Embeddings generados y guardados en CSV: {embeddings_path}")
    return embeddings_dict

def cargar_embeddings(csv_file):
    df = pd.read_csv(csv_file)
    embeddings_dict = {}
    for _, row in df.iterrows():
        jugador = row['Player']
        embedding = row.iloc[1:].to_numpy(dtype=np.float32)
        embeddings_dict[jugador] = embedding
    return embeddings_dict

