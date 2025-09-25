
# Redes Neuronales desde Cero con PyTorch (Enfoque Matemático y Aplicación Deportiva)

Este proyecto tiene como objetivo implementar redes neuronales desde un punto de vista introductorio y matemático, mostrando cómo funcionan internamente los modelos y su entrenamiento usando únicamente tensores y funciones de PyTorch (sin `nn.Module`).

## Objetivo de la implementación

- Facilitar la comprensión de los fundamentos matemáticos y computacionales de las redes neuronales, implementando desde cero modelos como perceptrones multicapa (MLP) y autoencoders.
- Permitir visualizar y entender la estructura interna de las redes, guardando los modelos en formato `.json` para que su arquitectura y parámetros sean fácilmente inspeccionables.
- Proveer ejemplos claros y didácticos para estudiantes o cualquier persona interesada en los fundamentos antes de usar frameworks de alto nivel.

## Objetivo de la aplicación

Además de la parte didáctica, el proyecto incluye una aplicación práctica al análisis deportivo en baloncesto (NBA):

- Se utilizan autoencoders para aprender **embeddings** (representaciones latentes) informativos sobre los estilos de juego de los jugadores a partir de estadísticas reales de la NBA.
- Posteriormente, se agrupan estos embeddings en **clusters** para generar un "mapa de estilos de juego", donde jugadores con estilos similares aparecen próximos entre sí en el espacio latente.
- Esto permite analizar, visualizar y comparar estilos de juego de manera objetiva y exploratoria.

## Estructura y descripción de los ficheros principales

- **MLP.py**: Implementación desde cero de una red neuronal multicapa (MLP), con clases para capas y la red completa. Incluye métodos de entrenamiento, guardado y carga en JSON.
- **Autoencoder.py**: Define la clase `Autoencoder` usando dos MLP (encoder y decoder). Permite entrenar, guardar y cargar autoencoders personalizados.
- **MLP_crear.py** / **MLP_entrenar.py**: Scripts para crear y entrenar modelos MLP, configurando arquitectura, hiperparámetros y opciones de guardado.
- **Autoencoder_crear.py** / **Autoencoder_entrenamiento.py**: Scripts para crear y entrenar autoencoders, permitiendo definir la arquitectura y entrenar sobre diferentes tipos de datos.
- **Procesar_datos_avanzado.py** / **Procesar_datasets.py**: Funciones para procesar, normalizar y filtrar los datos de entrada (estadísticas NBA), devolviendo tensores listos para los modelos.
- **Embeddings.py**: Genera embeddings de jugadores usando el encoder de un autoencoder y los guarda en CSV/diccionario.
- **Clustering.py**: Aplica clustering (K-Means) sobre los embeddings o sobre los datos procesados, y visualiza los resultados en 2D (usando PCA o el encoder de un autoencoder).
- **main.py**: Orquesta el flujo completo: genera embeddings, agrupa en clusters y visualiza los resultados, permitiendo comparar el espacio latente aprendido con métodos clásicos como PCA.
- **Guardar_Cargar.py**: Funciones generales para guardar y cargar modelos de cualquier tipo (MLP, Autoencoder, Clasificador), detectando el tipo automáticamente.
- **Red_clasificador/**: Carpeta donde se está implementando una red de clasificador (aún en desarrollo) para, en el futuro, entrenar clasificadores sobre los embeddings generados.

## Organización de carpetas y recursos

- **datasets/**: Contiene datasets de la NBA de las temporadas 2018 a 2025, listos para ser usados en el entrenamiento y análisis con las redes.
- **redes_disponibles/**: Incluye algunos modelos ya entrenados (MLP y autoencoders) para pruebas y análisis directo.
- **embeddings/**: Contiene diccionarios de embeddings generados para diferentes configuraciones y temporadas.
- **clusterings/**: Incluye resultados de agrupaciones (clusters) ya generados, útiles para visualizar y comparar estilos de juego.
- **Red_clasificador/**: Carpeta dedicada al desarrollo de una red de clasificación sobre embeddings (en construcción).

## Requisitos

Instala las dependencias ejecutando:

```
pip install -r requirements.txt
```

## Notas finales
- El código está pensado para ser didáctico y transparente, ideal para quienes quieren aprender los fundamentos matemáticos y computacionales de las redes neuronales y ver una aplicación real en el análisis deportivo.
- Puedes explorar los resultados y modificar los scripts para experimentar con diferentes arquitecturas, datos y métodos de análisis.
