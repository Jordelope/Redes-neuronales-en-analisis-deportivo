from Autoencoder import Autoencoder
from Guardar_Cargar import cargar_modelo
from Embeddings import generar_embeddings
from Clustering import clustering_embeddings


## Nombres archivos ##

archivo_AE = r"redes_disponibles\finales\overcomplete\AE_over_shtgStatPG.json"  # archivo  del autoencoder
archivo_scout = r"datasets\nba\finales\scout_pergame_nba24_25.csv"            # CSV con datos de los jugadores
archivo_emb = r"embeddings\emb_scout24_25_shtgStatPG_overcomplete.csv"  # archivo CSV de embeddings
archivo_clust =  r"clusterings\cltr_scout24_25_shtgStatPG_overcomplete_5cltrs.csv" # archivo CSV clusters
archivo_graf = r"clusterings\graficos\img_scout24_25_shtgStatPG_overcomplete_5cltrs.png" #archivo imagen (.png o .pdf)

## Ajustes scouting ##

modo_stats = "tiro"
numero_clusters = 5

if __name__ == "__main__":

    dict_emb = generar_embeddings(archivo_AE,archivo_scout,archivo_emb,modo_stats,)
    
    dict_clust = clustering_embeddings(archivo_emb,numero_clusters,archivo_clust,guardar_grafico=True,archivo_grafico=archivo_graf)


