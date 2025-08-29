import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Embeddings import cargar_embeddings

player_embeddings = ""
archivo_clusters = ""
n_clusters = 5


def clustering_embeddings(archivo_embeddings, n_clusters=5, archivo_salida="clusters.csv", graficar=True, met_red_dim : str="pca"):

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
    if graficar:
        if met_red_dim == "pca":
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.7)
            plt.legend(*scatter.legend_elements(), title="Cluster")
            for i, jugador in enumerate(jugadores):
                plt.text(X_pca[i, 0]+0.02, X_pca[i, 1]+0.02, jugador, fontsize=8, alpha=0.6)
            plt.title("Clustering de jugadores (PCA 2D)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.show()

    return dicc_clusters

#----------------------------------------------------------------------------------------------------------------------------------------------

if __name__=="__main__":

    clustering_embeddings(player_embeddings,archivo_clusters,n_clusters)