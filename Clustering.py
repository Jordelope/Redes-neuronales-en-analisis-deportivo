"""
Clustering.py
-------------

Este script proporciona funciones para generar, guardar, cargar y visualizar clusters(grupos) de  jugadores de la NBA  a partir de embeddings generados por un autoencoder:

1. generar_clusters: Usando el algoritmo de k-medias agrupa embeddings segun su proximidad en el espacio latente.
2. guardar_clusters y cargar_clusters: Permite guardar y cargar los diccionarios con los clusters usando ficheros de tipo csv.
3. visual_clusters: Usando un modelo de reduccion de dimensionalidad permite visualizar los embeddings en 2 o 3 dimensiones agrupando por colores segun unos clusters dados.

"""
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Autoencoder import Autoencoder
from Guardar_Cargar import cargar_modelo
from Embeddings import cargar_embeddings
from Procesar_datos_AE import procesar_datos




def generar_clusters(embeddings_dict:dict, n_clusters:int):
    # Preparar datos
    jugadores = list(embeddings_dict.keys())
    X = torch.tensor([embeddings_dict[j] for j in jugadores], dtype=torch.float32)

    # K-means sobre embeddings originales
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X.numpy())
    
    # ---- Guardar resultados en diccionario ---
    dicc_clusters = {jugador: int(cluster) for jugador, cluster in zip(jugadores, clusters)}
    return dicc_clusters

def guardar_clusters(dicc_clusters:dict, archivo_salida:str):
    # --- Guardar como csv --
    df_out = pd.DataFrame(list(dicc_clusters.items()), columns=["Player", "Cluster"])
    df_out.to_csv(archivo_salida, index=False)
    print(f"Clusters guardados en {archivo_salida}")

def cargar_clusters(archivo_clusters:str):
    df = pd.read_csv(archivo_clusters)
    dicc_clusters = {}
    for _,row in df.iterrows():
        jugador = row['Player']
        clt = int(row['Cluster'])
        dicc_clusters[jugador] =clt
    return dicc_clusters

def visual_clusters(embeddings_dict:dict, dicc_clusters:dict, n_clusters:int, modelo_visual:Autoencoder=None):
    
    # Extraer encoder del modelo
    modelo= cargar_modelo(modelo_visual)
    encoder = modelo.encoder

    # Preparar datos de embeddings
    jugadores = list(embeddings_dict.keys())
    clusters = list(dicc_clusters.values())
    
    # Transformamos embeddings a tensores para operar  
    X = torch.tensor([embeddings_dict[j] for j in jugadores], dtype=torch.float32)
    
    # Reducir dimensión con encoder para graficar
    with torch.no_grad():
        X_encoded = encoder(X).numpy()  # Se asume que encoder devuelve 2D
    
    # Gráfico
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=clusters, cmap="tab10", alpha=0.7)
    
    # Etiquetas de todos los jugadores
    for i, jugador in enumerate(jugadores):
        plt.text(X_encoded[i, 0] + 0.02, X_encoded[i, 1] + 0.02, jugador, fontsize=7)

    plt.title(f"Clusters sobre embeddings, graficados con encoder. (k={n_clusters})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()


#----------------------------------------------------------------------------------------------------------------------------

"""
Las siguientes funciones se encuentran desactualizadas y están pendientes de ser eliminadas.
"""

def clusterizar_pca(csv_file, tipo_stats="completo", n_clusters=5, n_components=2):
    """
    Procesa datos con procesar_datos, aplica PCA y agrupa con K-Means.
    Devuelve los clusters y muestra un scatter plot.
    """
    # Procesar datos con tu función
    X_tensor, etiquetas = procesar_datos(csv_file, tipo_stats=tipo_stats)

    # Convertir a numpy
    X = X_tensor.numpy()

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    # Visualización
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.7)

    # Añadir etiquetas de TODOS los jugadores
    for i, jugador in enumerate(etiquetas["Player"]):
        plt.text(X_pca[i, 0] + 0.02, X_pca[i, 1] + 0.02, jugador, fontsize=7)

    plt.title(f"Clusters sobre las estadisticas, graficados con PCA (k={n_clusters}, tipo_stats={tipo_stats})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()

def clustering_autoencoder(archivo_modelo, embeddings_path, archivo_salida, n_clusters=5):
    """
    Aplica clustering a embeddings usando K-means sobre los embeddings originales 
    y grafica las salidas reducidas a 2D por un encoder de autoencoder entrenado.
    
    archivo_modelo: ruta del archivo del autoencoder entrenado (cargado con cargar_modelo)
    embeddings_dict: dict {jugador: embedding (np.array o torch.Tensor)}
    n_clusters: número de clusters para K-means
    """
    
    embeddings_dict = cargar_embeddings(embeddings_path)

    # Cargar autoencoder y extraer encoder
    autoencoder = cargar_modelo(archivo_modelo)
    encoder = autoencoder.encoder

    # Preparar datos
    jugadores = list(embeddings_dict.keys())
    X = torch.tensor([embeddings_dict[j] for j in jugadores], dtype=torch.float32)

    # K-means sobre embeddings originales
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X.numpy())
    
    # ---- Guardar resultados en diccionario ---
    dicc_clusters = {jugador: int(cluster) for jugador, cluster in zip(jugadores, clusters)}

    # --- Guardar como csv --
    df_out = pd.DataFrame(list(dicc_clusters.items()), columns=["Player", "Cluster"])
    df_out.to_csv(archivo_salida, index=False)
    print(f"Clusters guardados en {archivo_salida}")


    # Reducir dimensión con encoder para graficar
    with torch.no_grad():
        X_encoded = encoder(X).numpy()  # Se asume que encoder devuelve 2D

    # Gráfico
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=clusters, cmap="tab10", alpha=0.7)
    
    # Etiquetas de todos los jugadores
    for i, jugador in enumerate(jugadores):
        plt.text(X_encoded[i, 0] + 0.02, X_encoded[i, 1] + 0.02, jugador, fontsize=7)

    plt.title(f"Clusters sobre embeddings, graficados con encoder. (k={n_clusters})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()
