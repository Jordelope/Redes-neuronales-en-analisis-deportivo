from Autoencoder import Autoencoder
from Guardar_Cargar import cargar_modelo
from Embeddings import generar_embeddings
from Clustering import clustering_embeddings, graficar_clusters

## Nombres archivos ##

archivo_emb = r"embeddings\emb_scout24_25_shtgStatPG_overcomplete.csv"  # archivo CSV de embeddings
archivo_clust =  r"clusterings\cltr_scout24_25_shtgStatPG_overcomplete_5cltrs.csv" # archivo CSV clusters
archivo_graf = r"clusterings\graficos\img_scout24_25_shtgStatPG_overcomplete_5cltrs.png" #archivo imagen (.png o .pdf)

## Ajustes scouting ##

modo_stats = "tiro"
numero_clusters = 5

if __name__ == "__main__":

    graficar_clusters(archivo_clust,archivo_emb)