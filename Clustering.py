import torch
import pandas as pd
import numpy as np
from Autoencoder import Autoencoder
from Guardar_Cargar import cargar_modelo
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Embeddings import cargar_embeddings
from Procesar_datos_avanzado import procesar_datos

player_embeddings = r"embeddings\emb_scout24_25_shtgStatPG_overcomplete.csv"
archivo_clusters = r"clusterings\cltr_scout24_25_shtgStatPG_overcomplete_10cltrs.csv"
archivo_grafico = r"clusterings\graficos\red_dim_ae.png"
autoencoder_redDim = r"redes_disponibles\finales\AE_redDim2_dimIN80_.json"

n_clusters = 10


def clustering_embeddings(archivo_embeddings, n_clusters=5, archivo_salida="clusters.csv", guardar_grafico=False, archivo_grafico="clusters.png", met_red_dim : str="pca"):

    # --- Cargar embeddings ---
    embeddings_dict = cargar_embeddings(archivo_embeddings)
    jugadores = list(embeddings_dict.keys())
    X = np.array(list(embeddings_dict.values()))

    # --- KMeans clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # ---- Guardar resultados en diccionario ---
    dicc_clusters = {jugador: int(cluster) for jugador, cluster in zip(jugadores, labels)}

    # --- Guardar como csv --
    df_out = pd.DataFrame(list(dicc_clusters.items()), columns=["Player", "Cluster"])
    df_out.to_csv(archivo_salida, index=False)
    print(f"Clusters guardados en {archivo_salida}")

    # ---- Visualización opcional ----
    if guardar_grafico:
        if met_red_dim == "pca":
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)
            ax.legend(*scatter.legend_elements(), title="Cluster")
            for i, jugador in enumerate(jugadores):
                ax.text(X_pca[i, 0]+0.02, X_pca[i, 1]+0.02, jugador, fontsize=8, alpha=0.6)
            ax.set_title("Clustering de jugadores (PCA 2D)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.show()

            # Guardar gráfico
            fig.savefig(archivo_grafico, bbox_inches="tight")
            plt.close(fig)  
            print(f"Gráfico guardado en {archivo_grafico}")

    return dicc_clusters



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


def clustering_autoencoder(archivo_modelo, embeddings_path, n_clusters=5):
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

def graficar_clusters(archivo_clusters, archivo_embeddings, archivo_grafico=None):
    """
    Grafica los clusters y embeddings con PCA 2D y evita solapamiento de etiquetas.

    clusters_dict: dict {jugador: cluster}
    embeddings_dict: dict {jugador: embedding (np.array o list)}
    archivo_grafico: si no es None, guarda el gráfico (formato según extensión: .png, .svg, .pdf)
    """
    from adjustText import adjust_text

    # Cargar clusters
    df_clusters = pd.read_csv(archivo_clusters)
    clusters_dict = dict(zip(df_clusters['Player'], df_clusters['Cluster']))

    # Cargar embeddings
    df_embeddings = pd.read_csv(archivo_embeddings)
    embeddings_dict = {row['Player']: row.iloc[1:].to_numpy(dtype=np.float32) 
                       for _, row in df_embeddings.iterrows()}

    jugadores = list(clusters_dict.keys())
    labels = np.array([clusters_dict[j] for j in jugadores])
    X = np.array([embeddings_dict[j] for j in jugadores])

    # Reducción a 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)
    ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.set_title("Clusters de jugadores (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Etiquetas con ajuste automático
    texts = []
    for i, jugador in enumerate(jugadores):
        texts.append(ax.text(X_pca[i, 0], X_pca[i, 1], jugador, fontsize=8, alpha=0.7))
    
    # Mostrar
    plt.show()

    # Guardar si se pide
    if archivo_grafico:
        fig.savefig(archivo_grafico, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico guardado en {archivo_grafico}")
#----------------------------------------------------------------------------------------------------------------------------------------------

if __name__=="__main__":

    clustering_autoencoder(autoencoder_redDim,player_embeddings,10)
    clusterizar_pca(r"datasets\nba\finales\scout_pergame_nba24_25.csv","tiro",10)