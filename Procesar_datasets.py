import pandas as pd
import re


def filtrar_repetidos(df):
        resultado = []
        if "Team" in df.columns:
            for nombre, grupo in df.groupby("Player-additional"):
                total_row_2 = grupo[grupo["Team"] == "2TM"]
                total_row_3 = grupo[grupo["Team"] == "3TM"] # Revisar esto no falle
                total_row_4 = grupo[grupo["Team"] == "4TM"] # Revisar esto no falle
                if not total_row_2.empty:
                    resultado.append(total_row_2)
                elif not total_row_3.empty:
                    resultado.append(total_row_3)
                elif not total_row_4.empty:
                    resultado.append(total_row_4)
                else:
                    resultado.append(grupo.iloc[:1])
            return pd.concat(resultado)
        else:
            return df


def combinar2_datasets(dataset1, dataset2, salida):
    """
    Combina dos datasets CSV en uno solo sin eliminar duplicados.
    
    Parámetros:
    - dataset1 (str): ruta al primer CSV
    - dataset2 (str): ruta al segundo CSV
    - salida (str): ruta donde guardar el CSV combinado
    """
    # Leer ambos datasets
    df1 = pd.read_csv(dataset1)
    df2 = pd.read_csv(dataset2)

    # Concatenar sin eliminar duplicados
    combinado = pd.concat([df1, df2], ignore_index=True)

    # Guardar en un nuevo archivo
    combinado.to_csv(salida, index=False)

    return combinado



def completar_datasets(dataset1, dataset2, salida, clave="Player-additional",primera_fila=0):
    """
    Combina dos datasets de la misma temporada (mismos jugadores, diferentes estadísticas).
    - Conserva las columnas del primer dataset en caso de duplicados.
    - Une por la columna clave (por defecto 'Player-additional').
    - Emite avisos si se eliminaron columnas duplicadas.
    
    Parámetros:
    - dataset1 (str): ruta del primer CSV (columna principal).
    - dataset2 (str): ruta del segundo CSV (estadísticas adicionales).
    - salida (str): ruta donde guardar el dataset combinado.
    - clave (str): columna clave para emparejar jugadores (default: 'Player-additional').
    """
    
    # Cargar datasets
    df1 = pd.read_csv(dataset1)
    df2 = pd.read_csv(dataset2,header=primera_fila)  # salta la primera fila de categorías

    # Normalizar nombres de clave (por si acaso)
    if clave not in df1.columns:
        raise ValueError(f"'{clave}' no está en {dataset1}")
    if clave not in df2.columns:
        raise ValueError(f"'{clave}' no está en {dataset2}")

    # Detectar columnas duplicadas
    columnas_comunes = [c for c in df1.columns if c in df2.columns and c != clave and c != "Team"]
    if columnas_comunes:
        print(f"⚠ Aviso: Se encontraron columnas duplicadas {columnas_comunes}. "
              f"Se conservarán las del primer dataset ({dataset1}).")
        # Eliminar duplicadas del segundo dataset
        df2 = df2.drop(columns=columnas_comunes)

    # Hacer merge en base a la clave
    print("Columnas de df2:", df2.columns.tolist())
    combinado = pd.merge(df1, df2, on=[clave,"Team"], how="inner")

    # Guardar en CSV
    combinado.to_csv(salida, index=False)
    print(f"Se ha completado {dataset1} con {dataset2} en el nuevo {salida}.")
    return combinado





def combinar_varios_datasets_filtrando(lista_datasets, salida, clave="Player-additional", temporada_tag=True):
    """
    Combina varios datasets CSV en uno solo sin eliminar duplicados entre temporadas,
    pero dentro de cada dataset filtra a los jugadores con múltiples equipos,
    dejando solo la fila con '2TM', '3TM', '4TM', etc.
    Si temporada_tag=True, añade el identificador de temporada al final de la clave 
    para distinguir temporadas (extraído del nombre de archivo).
    
    Parámetros:
    - lista_datasets (list): lista con las rutas de los CSV a combinar
    - salida (str): ruta donde guardar el CSV combinado
    - clave (str): columna clave del jugador (default: 'Player-additional')
    - temporada_tag (bool): si True, añade sufijo con temporada
    """
    dataframes = []

    for archivo in lista_datasets:
        df = pd.read_csv(archivo)

        # Extraer temporada del nombre del archivo: busca algo tipo "ba19_20"
        temporada_match = re.search(r"ba\d{2}_\d{2}", archivo)
        temporada = temporada_match.group(0) if temporada_match else archivo.split("/")[-1].split(".")[0]

        # Modificar clave para distinguir temporadas
        if temporada_tag:
            df[clave] = df[clave].astype(str) + f"_{temporada}"

        # Filtrar duplicados de jugadores en varios equipos:
        filtrado = []
        for jugador, grupo in df.groupby("Player", sort=False):
            multi_team = grupo[grupo["Team"].str.contains("TM", na=False)]
            if not multi_team.empty:
                filtrado.append(multi_team.iloc[0].to_dict())  # Guardamos la fila nTM
            else:
                filtrado.extend(grupo.to_dict("records"))  # Guardamos todas las demás filas

        df_filtrado = pd.DataFrame(filtrado)

        # Reordenar por Rk (manteniendo orden profesional)
        if "Rk" in df_filtrado.columns:
            df_filtrado = df_filtrado.sort_values("Rk", ascending=True)

        dataframes.append(df_filtrado)

    # Concatenar todos los datasets ya filtrados (se respeta el orden en la lista)
    combinado = pd.concat(dataframes, ignore_index=True)

    # Guardar en CSV
    combinado.to_csv(salida, index=False)

    return combinado


#---------------------------------------------------------------------------------------------------------------------------
lista_datsets_trad_per36 = [r"datasets\nba\per36\nba18_19_trad.csv",
                            r"datasets\nba\per36\nba19_20_trad.csv",   # Datasets trad per36
                            r"datasets\nba\per36\nba20_21_trad.csv",
                            r"datasets\nba\per36\nba21_22_trad.csv",
                            r"datasets\nba\per36\nba22_23_trad.csv",
                            r"datasets\nba\per36\nba23_24_trad.csv",
                            r"datasets\nba\per36\nba24_25_trad.csv"
                    ]
lista_datsets_trad_per100 = [r"datasets\nba\per100\nba18_19_trad.csv",
                             r"datasets\nba\per100\nba19_20_trad.csv",   # Datasets trad per 100
                             r"datasets\nba\per100\nba20_21_trad.csv",
                             r"datasets\nba\per100\nba21_22_trad.csv",
                             r"datasets\nba\per100\nba22_23_trad.csv",
                             r"datasets\nba\per100\nba23_24_trad.csv",
                             r"datasets\nba\per100\nba24_25_trad.csv"
                    ]
lista_datsets_trad_pergame = [r"datasets\nba\pergame\nba18_19_trad.csv",
                              r"datasets\nba\pergame\nba19_20_trad.csv",   # Datasets trad pergame
                              r"datasets\nba\pergame\nba20_21_trad.csv",
                              r"datasets\nba\pergame\nba21_22_trad.csv",
                              r"datasets\nba\pergame\nba22_23_trad.csv",
                              r"datasets\nba\pergame\nba23_24_trad.csv",
                              r"datasets\nba\pergame\nba24_25_trad.csv"
                    ]

lista_datasets_shtg = [ r"datasets\nba\pergame\nba18_19_shooting.csv",
                        r"datasets\nba\pergame\nba19_20_shooting.csv",   # Datasets shooting
                        r"datasets\nba\pergame\nba20_21_shooting.csv",
                        r"datasets\nba\pergame\nba21_22_shooting.csv",
                        r"datasets\nba\pergame\nba22_23_shooting.csv",
                        r"datasets\nba\pergame\nba23_24_shooting.csv",
                        r"datasets\nba\pergame\nba24_25_shooting.csv"
                        ]  
lista_datsets_adv = [r"datasets\nba\pergame\nba18_19_advanced.csv",
                     r"datasets\nba\pergame\nba19_20_advanced.csv",   # Datasets advanced
                     r"datasets\nba\pergame\nba20_21_advanced.csv",
                     r"datasets\nba\pergame\nba21_22_advanced.csv",
                     r"datasets\nba\pergame\nba22_23_advanced.csv",
                     r"datasets\nba\pergame\nba23_24_advanced.csv",
                     r"datasets\nba\pergame\nba24_25_advanced.csv"
                    ] 

archivos_salida_temporadas_per100 = [r"datasets\nba\per100\nba18_19_completo.csv",
                                     r"datasets\nba\per100\nba19_20_completo.csv",   # Datasets 100
                                     r"datasets\nba\per100\nba20_21_completo.csv",
                                     r"datasets\nba\per100\nba21_22_completo.csv",
                                     r"datasets\nba\per100\nba22_23_completo.csv",
                                     r"datasets\nba\per100\nba23_24_completo.csv",
                                     r"datasets\nba\per100\nba24_25_completo.csv"
                    ]
archivos_salida_temporadas_per36 = [r"datasets\nba\per36\nba18_19_completo.csv",
                                    r"datasets\nba\per36\nba19_20_completo.csv",   # Datasets 36
                                    r"datasets\nba\per36\nba20_21_completo.csv",
                                    r"datasets\nba\per36\nba21_22_completo.csv",
                                    r"datasets\nba\per36\nba22_23_completo.csv",
                                    r"datasets\nba\per36\nba23_24_completo.csv",
                                    r"datasets\nba\per36\nba24_25_completo.csv"
                    ]
archivos_salida_temporadas_pergame = [r"datasets\nba\pergame\nba18_19_completo.csv",
                                      r"datasets\nba\pergame\nba19_20_completo.csv",   # Datasets pergame
                                      r"datasets\nba\pergame\nba20_21_completo.csv",
                                      r"datasets\nba\pergame\nba21_22_completo.csv",
                                      r"datasets\nba\pergame\nba22_23_completo.csv",
                                      r"datasets\nba\pergame\nba23_24_completo.csv",
                                      r"datasets\nba\pergame\nba24_25_completo.csv"
                    ]


archivo_entrenamiento_completo_per100 = r"datasets\nba\finales\nba19_24_per100_entrenamiento.csv"

archivo_entrenamiento_completo_per36 = r"datasets\nba\finales\nba19_24_per36_entrenamiento.csv"  

archivo_entrenamiento_completo_pergame = r"datasets\nba\finales\nba19_24_pergame_entrenamiento.csv"

archivo_test_per100 = r"datasets\nba\finales\test_per100_nba18_19.csv"
archivo_test_per36 = r"datasets\nba\finales\test_per36_nba18_19.csv"
archivo_test_pergame = r"datasets\nba\finales\test_pergame_nba18_19.csv"

archivo_scout_per100 = r"datasets\nba\finales\scout_per100_nba24_25.csv"
archivo_scout_per36 = r"datasets\nba\finales\scout_per36_nba24_25.csv"
archivo_scout_pergame = r"datasets\nba\finales\scout_pergame_nba24_25.csv"

combinar_finales_total = True
completar_temporadas = True



if __name__=="__main__":
    
    
    if completar_temporadas :
        
        for i in range(len(archivos_salida_temporadas_per100)):
            completar_datasets(lista_datsets_trad_per100[i] , lista_datsets_adv[i] , archivos_salida_temporadas_per100[i])
            
            if i == 0:
                completar_datasets(archivos_salida_temporadas_per100[i] , lista_datasets_shtg[i] , archivo_test_per100,"Player-additional",1)
            
            elif i== len(archivos_salida_temporadas_per100)-1:
                completar_datasets(archivos_salida_temporadas_per100[i] , lista_datasets_shtg[i] , archivo_scout_per100,"Player-additional",1)
            
            else:
                completar_datasets(archivos_salida_temporadas_per100[i] , lista_datasets_shtg[i] , archivos_salida_temporadas_per100[i],"Player-additional",1)
        #------------------
        for i in range(len(archivos_salida_temporadas_per36)):
            completar_datasets(lista_datsets_trad_per36[i] , lista_datsets_adv[i] , archivos_salida_temporadas_per36[i])
            
            if i == 0:
                completar_datasets(archivos_salida_temporadas_per36[i] , lista_datasets_shtg[i] , archivo_test_per36,"Player-additional",1)

            elif i == len(archivos_salida_temporadas_per36)-1:
                completar_datasets(archivos_salida_temporadas_per36[i] , lista_datasets_shtg[i] , archivo_scout_per36,"Player-additional",1)

            else:
                completar_datasets(archivos_salida_temporadas_per36[i] , lista_datasets_shtg[i] , archivos_salida_temporadas_per36[i],"Player-additional",1)
        #-------------------
        for i in range(len(archivos_salida_temporadas_pergame)):
            completar_datasets(lista_datsets_trad_pergame[i] , lista_datsets_adv[i] , archivos_salida_temporadas_pergame[i])
            
            if i == 0:
                completar_datasets(archivos_salida_temporadas_pergame[i] , lista_datasets_shtg[i] , archivo_test_pergame,"Player-additional",1)
            
            elif i == len(archivos_salida_temporadas_pergame)-1 :
                completar_datasets(archivos_salida_temporadas_pergame[i] , lista_datasets_shtg[i] , archivo_scout_pergame,"Player-additional",1)

            else:
                completar_datasets(archivos_salida_temporadas_pergame[i] , lista_datasets_shtg[i] , archivos_salida_temporadas_pergame[i],"Player-additional",1)


    combinar_varios_datasets_filtrando(archivos_salida_temporadas_per100[1:-1],archivo_entrenamiento_completo_per100)
    combinar_varios_datasets_filtrando(archivos_salida_temporadas_per36[1:-1],archivo_entrenamiento_completo_per36)
    combinar_varios_datasets_filtrando(archivos_salida_temporadas_pergame[1:-1],archivo_entrenamiento_completo_pergame)

    