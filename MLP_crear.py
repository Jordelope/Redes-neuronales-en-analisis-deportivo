"""
MLP_crear.py
------------

Este script está diseñado para crear nuevos modelos de perceptrón multicapa (MLP) utilizando la clase MLP implementada desde cero en este proyecto.
Permite definir la arquitectura de la red (número de entradas, salidas, capas ocultas y funciones de activación), instanciar el modelo y guardarlo en disco para su posterior entrenamiento o uso.

Funcionalidad principal:
- Configuración de la arquitectura del modelo MLP.
- Instanciación de la red con los parámetros definidos.
- Opción de añadir una descripción personalizada al modelo.
- Guardado del modelo en un archivo especificado.

Uso:
1. Ajusta los parámetros de arquitectura y el nombre del archivo donde se guardará la red.
2. Ejecuta el script para crear y guardar el modelo MLP.
3. El modelo podrá ser cargado y entrenado posteriormente con otros scripts del proyecto.
"""


import torch
import torch.nn.functional as F
import random
from MLP import MLP
from Guardar_Cargar import guardar_modelo


## Nombre del nuevo modelo ##

nombre_archivo_red = r""  # Archivo donde se guarda la red


## ARQUITECTURA de la red ## (Prestar MUCHA ATENCION A FUNCIONES ACTIVACION)
input_sz = 18             # Número de entradas (asegurar que coincide con los datos que se va a usar)
out_sz = 5                # Número de salidas (asegurar que coincide con los datos que se va a usar)
estructura_oct = []       # Capas ocultas
lista_act = None          # Lista de funciones de activación (asegurar compatible con estructura oct)
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
