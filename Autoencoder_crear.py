import torch
import torch.nn.functional as F
import random
from MLP import MLP
from Autoencoder import Autoencoder
from Guardar_Cargar import guardar_modelo, cargar_modelo
from Procesar_datos import procesar_datos


## NOMBRE archivos (se tomaran los MLP si ya los tenemos) ##
existen_MLP = False
archivo_encod = r"" 
archivo_decod = r"" 
archivo_autoencoder = r"redes_disponibles\finales\undercomplete\AE_under_shtgStatPG.json" 


## OPCIONES de entrenado y guardado ##
save_autoencoder = True
save_decoder = False
save_encoder = False

añadir_descripcion = True  # Opción de añadir una descripción
descripcion = " Autoencoder undercomplete para tratar estadisticas tiro en per game."

## ESTRUCTURA autoencoder (si no tenemos los MLP) ##

input_sz = 30            # Número de entradas
lat_spc_dim = 6         # Dimension espacio latente(salida encoder, entrada decoder) 

estructura_encod = [64, 16]               # Capas ocultas encoder
estructura_decod = estructura_encod[::-1]      # Capas ocultas decoder

lista_act_encod = [F.leaky_relu , F.elu, F.leaky_relu]  # Funciones activacion encoder (None = [None,...,None] por defecto lineal en MLP)
lista_act_decod = [ F.elu, F.leaky_relu] + [None] # Funciones activacion encoder (None = [None,...,None] por defecto lineal en MLP)


if __name__ == "__main__":
    
    if not save_autoencoder and not save_encoder and not save_decoder:
        print("AVISO: No se guardará ninguna red.\n")
    

    # Cargamos los MLP o los creamos si no existen
    if existen_MLP:
        # Cargar MLP existentes
        encoder = cargar_modelo(archivo_encod)
        decoder = cargar_modelo(archivo_decod)
    else:
        # Crear nuevos MLP para encoder y decoder
        encoder = MLP(input_sz, lat_spc_dim, estructura_encod, lista_act_encod)
        decoder = MLP(lat_spc_dim, input_sz, estructura_decod, lista_act_decod)
    
    
    # Crear autoencoder
    autoencoder = Autoencoder(encoder, decoder)
    

    # Guardamos las redes segun eleccion
    if añadir_descripcion:
        autoencoder.description = descripcion
    if save_autoencoder:
        guardar_modelo(autoencoder, archivo_autoencoder)
        if save_encoder :
            guardar_modelo(encoder, archivo_encod)
        if save_decoder:
            guardar_modelo(decoder, archivo_decod) 
    else:
        print(f"Se ha decidido no guardar la red {archivo_autoencoder}.\n")

