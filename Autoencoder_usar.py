import torch
import torch.nn.functional as F
import numpy
from MLP import MLP, nombre_a_func
from Autoencoder import Autoencoder 
from Clasificador import Clasificador 
from Guardar_Cargar import guardar_modelo, cargar_modelo
from Procesar_datos import procesar_datos
from Visual import visual



## Funciones_relevantes ##
def vector_a_clase(xs):
    return xs.argmax().item()

## NOMBRE Clasificador ##
archivo_modelo = r"redes_disponibles/autoencoder_nba2_mejoras.json"
ver_descripcion = True

## OPCIONES visualizacion ##
reducir_dimension = True

dim_repr = 2 
modos_redd_dim = ["pca", "tsne", "umap"]
modo_redd_dim = "pca"
visualizar_todos = False
titulo_grafico = "Titulo"

## DATOS a clasificar / TEST a evaluar ##

xs_a_clasificar = None
ys_para_evaluar = None

archivo_entrenamiento = r"datasets\nba\combined19_25_pergame_filtered.csv"
archivo_test = r"datasets\nba\nba24_25_pergame.csv"
_, _, _, xs_a_clasificar, ys_para_evaluar, etiquetas_test = procesar_datos(archivo_set_train=archivo_entrenamiento,
                                                                                    archivo_set_test=archivo_test,
                                                                                    modo_autoencoder=True,
                                                                                    modo_columnas="solo_volumen",
                                                                                    modo_targets="pos",
                                                                                    modo_etiquetado="posicion",
                                                                                    normalizar_datos=True,
                                                                                    modo_normalizacion="zscore",
                                                                                    umbral_partidos=20,
                                                                                    umbral_minutos=15,
                                                                                    umbral_en_test=True,
                                                                                    hay_fila_total_entrenamiento=False,
                                                                                    hay_fila_total_test=True)




#------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__=="__main__":
    
    # Cargamos Clasificador
    print(f"Se va a usar el Clasificador: {archivo_modelo}.\n")
    autoencoder = cargar_modelo(archivo_modelo)
    if ver_descripcion:
        print(f"La descripcion de este modelo nos dice: \n'{autoencoder.description}'")

    
    # Pasamos datos por el clasificador y extraemos su clase
    encoded_data = autoencoder.encoder(xs_a_clasificar).detach().numpy()


    # Visualización
    if reducir_dimension:
        if visualizar_todos:
            visual(archivo_modelo,xs_a_clasificar,dim_repr,etiquetas_test,titulo_grafico,"pca")
            
            visual(archivo_modelo,xs_a_clasificar,dim_repr,etiquetas_test,titulo_grafico,"tsne")
           
            visual(archivo_modelo,xs_a_clasificar,dim_repr,etiquetas_test,titulo_grafico,"umap")
            

        else:
            visual(archivo_modelo,xs_a_clasificar,dim_repr,etiquetas_test,titulo_grafico,modo_redd_dim)
            

    


