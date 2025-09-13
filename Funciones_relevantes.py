import torch
import torch.nn.functional as F
import random


"""
Este fichero contiene funciones que se emplean en la implementación de las redes 
neuronales y su aplicación al análisis deportivo.
"""




def get_batches(Xs : list, Ys : list, batch_size : int): 
    """
    Genera lotes aleatorios de datos para entrenamiento por lotes (batch training).
    - Xs: lista de entradas.
    - Ys: lista de salidas/targets.
    - batch_size: tamaño del lote.
    Devuelve tuplas (X_batch, Y_batch) de tamaño batch_size.
    """
    n = len(Xs)
    indices = list(range(n))
    random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start+batch_size]
        yield [Xs[i] for i in batch_idx], [Ys[i] for i in batch_idx]

nombre_a_func = {
    "relu": torch.relu,
    "softmax": F.softmax,
    "log_softmax": F.log_softmax,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
    "cross_entropy": F.cross_entropy,
    "binary_cross_entropy": F.binary_cross_entropy,
    "mse_loss": F.mse_loss,
    "none": None
}

def clasificacion(xs):# Devuelve el índice de la clase con mayor probabilidad
    """ 
    Devuelve el índice del valor mas alto. 
    En clasificación el indice de la clase con mayor probabilidad. 
    """
    return xs.argmax().item()  




