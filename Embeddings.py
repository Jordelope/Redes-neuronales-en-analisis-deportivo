import torch
import pandas as pd
import numpy as np
from MLP import MLP
from Autoencoder import Autoencoder
from Guardar_Cargar import cargar_modelo
from Procesar_datos import procesar_datos, filtrar_repetidos

# ---------- Nombre archivos ----------
autoencoder_path = "autoencoder.pth"  # archivo  del autoencoder
players_path = "jugadores.csv"            # CSV con datos de los jugadores
embeddings_path = "embeddings_dict.csv"  # archivo CSV de salida



def generar_embeddings( autoencoder_path,
                        players_path,
                        embeddings_path,
                        modo_columnas: str="solo_volumen",
                        modo_normalizacion: str="zscore"
                        ):
    
    # ---------- Cargar modelo y procesar datos ----------
    autoencoder = cargar_modelo(autoencoder_path)
    df = pd.read_csv(players_path)
    df_filtro_repes = filtrar_repetidos(df)
    player_ids = df_filtro_repes['Player'].values  # identificador de cada jugador
    datos_jugadores,_,_,_,_,_ = procesar_datos(players_path,players_path,True,modo_columnas,modo_normalizacion=modo_normalizacion,umbral_partidos=0,umbral_minutos=0)
    
    # ---------- GENERAR EMBEDDINGS ----------
    with torch.no_grad():
        embeddings_tensor = autoencoder.encoder(datos_jugadores)  # shape (num_jugadores, latent_size)
        embeddings_np = embeddings_tensor.numpy()  # convertimos todo a NumPy

    # Creamos dicc jugador: embedding
    embeddings_dict = {jugador: emb for jugador, emb in zip(player_ids, embeddings_np)}

    # ---------- GUARDAR EN CSV ----------
    embeddings_list = [[jugador] + embeddings_dict[jugador].tolist() for jugador in player_ids]
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


if __name__=="__main__":
    generar_embeddings(autoencoder_path,players_path,embeddings_path)