import torch
import torch.nn.functional as F
import random
from MLP import MLP
from Guardar_Cargar import guardar_modelo
from Procesar_datos import procesar_datos
"""
crear_red_torch.py
-------------------

Este script permite crear, entrenar y guardar una red neuronal multicapa (MLP) utilizando la implementación matemática definida en el módulo MLP.py.
El usuario puede definir la arquitectura de la red, los parámetros de entrenamiento y decidir si desea entrenar y/o guardar la red resultante en un archivo JSON.

Estructura general del script:
- Carga los datos de entrenamiento y test desde procesar_datos_entrenamiento.py.
- Permite definir la arquitectura y funciones de activación de la red MLP.
- Permite entrenar la red con los parámetros elegidos (número de pasos, tamaño de batch, función de pérdida, etc).
- Permite guardar la red (estructura y pesos) en un archivo JSON para su posterior uso.
- Incluye utilidades para decodificar la salida de la red y evaluar su precisión sobre el conjunto de test.

"""

## Nombre del nuevo modelo ##

nombre_archivo_red = r"redes_disponibles\mlp_prueba_desc.json"  # Archivo donde se guarda la red


## ARQUITECTURA de la red ## (Prestar MUCHA ATENCION A FUNCIONES ACTIVACION)
input_sz = 18      # Número de entradas (asegurar que coincide con los datos que se va a usar)
out_sz = 5        # Número de salidas (asegurar que coincide con los datos que se va a usar)
estructura_oct = [18, 18]  # Capas ocultas
lista_act = None           # Lista de funciones de activación (asegurar compatible con estructura oct)
                           #(None =[None,...,None] por defecto en MLP)


## OPCIONES GUARDADO  ##
save_new_NN = True         # ¿Guardar la red tras crearla?

añadir_descripcion = False  # Opción de añadir una descripción
descripcion = ""



#-------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    # Instanciación de la red
    NN = MLP(input_sz, out_sz, estructura_oct, lista_act)
    print(f"\nHemos creado la red {nombre_archivo_red}.\n")

    if save_new_NN:
        print(f"\nAVISO: La red {nombre_archivo_red} se va a guardar.\n")
    else:
        print(f"\nAVISO: La red {nombre_archivo_red} no se va a guardar.\n")

    # Guardado de la red
    if añadir_descripcion:
        NN.description = descripcion
    if save_new_NN:
        guardar_modelo(NN, nombre_archivo_red)
    else:
        print(f"Se ha decidido no guardar la red {nombre_archivo_red}.\n")
