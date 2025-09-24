"""
main.py
-------

Este script ejecuta el flujo principal de la aplicación: toma un autoencoder entrenado para generar embeddings de jugadores de la NBA, utiliza otro autoencoder para reducir la dimensionalidad de esos embeddings y aplica clustering para analizar y visualizar los resultados.

Pasos principales:
1. Genera un diccionario de embeddings de jugadores usando el encoder de un autoencoder.
2. Aplica clustering sobre los embeddings y reduce la dimensión para visualización usando un segundo autoencoder.
3. Guarda los resultados de los clusters y muestra los gráficos correspondientes.
4. (Opcional) Contrasta los resultados aplicando clustering y reducción de dimensión con PCA sobre los datos originales procesados.

Este flujo permite comparar la agrupación y representación de los jugadores en el espacio latente aprendido frente a métodos clásicos como PCA, facilitando el análisis exploratorio y visual de los datos.
"""


from Embeddings import generar_embeddings
from Clustering import clustering_autoencoder, clusterizar_pca


## Nombres archivos ##

archivo_AE        = r"redes_disponibles\finales\overcomplete\AE_over_shtgStatPG.json"            # archivo del autoencoder que genera embeddings
archivo_AE_redDim = r"redes_disponibles\finales\AE_redDim2_dimIN80_.json"                        # archivo del autoencoder que genera representación visual
archivo_scout     = r"datasets\nba\finales\scout_pergame_nba24_25.csv"                           # CSV con datos de los jugadores a analizar
archivo_emb       = r"embeddings\emb_scout24_25_shtgStatPG_overcomplete.csv"                     # archivo CSV donde guardar embeddings
archivo_clust     = r"clusterings\cltr_scout24_25_shtgStatPG_overcomplete_10cltrs.csv"           # archivo CSV donde guardar clusters


## Ajustes scouting ##

modo_stats = "tiro"
numero_clusters = 10       # Numero de clusters en los que agrupar

contrastar_con_pca = True # En caso True: Se aplica el metodo de contraste sobre los datos originales procesados con pca

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    ## Generar diccionario embeddings ##
    dict_emb = generar_embeddings(archivo_AE,archivo_scout,archivo_emb,modo_stats)
    
    ## Generar clusters
    dict_clust = clustering_autoencoder(archivo_AE_redDim, archivo_emb, archivo_clust, numero_clusters)
    
    if contrastar_con_pca: 
        clusterizar_pca(archivo_scout,modo_stats,numero_clusters,n_components=2)


