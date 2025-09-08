import pandas as pd
import numpy as np
import torch

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

def procesar_datos(csv_file, tipo_stats="completo"):
    df = pd.read_csv(csv_file)
    df = filtrar_repetidos(df)

    # Guardamos etiquetas (no se normalizan)
    etiquetas = df[["Player", "Pos", "Player-additional"]].copy()

    # Eliminamos columnas irrelevantes o duplicadas
    drop_cols = [
        "Rk", "Team", "Awards", "Player-additional",
        "2P.1", "3P.1",  # duplicadas de 2P% y 3P%
        "Att.", "Md."   # tiros de medio campo irrelevantes
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ------------------- Agrupación de columnas -------------------
    volumen_cols = ["G", "GS", "MP", "FGA", "FG", "3PA", "3P", "2PA", "2P",
                    "FTA", "FT", "ORB", "DRB", "TRB", "AST", "STL", "BLK",
                    "TOV", "PF", "PTS", "#"]

    porcentajes_cols = ["FG%", "3P%", "2P%", "FT%", 
                        "0-3.1", "3-10.1", "10-16.1", "16-3P.1"]

    eficiencia_cols = ["eFG%", "TS%", "PER"]

    ratios_cols = ["3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%",
                   "TOV%", "USG%", "WS/48", "%FGA", "%3PA", "2P.2", "3P.2"]

    plus_minus_cols = ["OBPM", "DBPM", "BPM", "VORP"]

    dist_cols = ["Dist.", "0-3", "3-10", "10-16", "16-3P"]

    ws_cols = ["OWS", "DWS", "WS"]

    # ------------------- Transformaciones -------------------
    for c in volumen_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c])
            df[c] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

    for c in porcentajes_cols:
        if c in df.columns:
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min() + 1e-8)

    for c in ratios_cols + ws_cols:
        if c in df.columns:
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min() + 1e-8)

    for c in plus_minus_cols:
        if c in df.columns:
            df[c] = (df[c] - df[c].mean()) / df[c].std(ddof=0)

    # ------------------- Selección según tipo_stats -------------------
    if tipo_stats == "completo":
        selected_cols = volumen_cols + porcentajes_cols + eficiencia_cols + \
                        ratios_cols + plus_minus_cols + dist_cols + ws_cols
    elif tipo_stats == "volumen":
        selected_cols = volumen_cols
    elif tipo_stats == "tiro":
        selected_cols = (
            # volumen relevante a tiro
            ["FGA", "FG", "3PA", "3P", "2PA", "2P", "FTA", "FT", "PTS"] +
            # acierto
            porcentajes_cols +
            # eficiencia (sin PER)
            ["eFG%", "TS%"] +
            # ratios relevantes
            ["3PAr", "FTr", "%FGA", "%3PA", "2P.2", "3P.2"] +
            # distancias
            dist_cols
        )
    else:
        raise ValueError(f"tipo_stats '{tipo_stats}' no válido. Usa: 'completo', 'volumen' o 'tiro'.")

    # Quitamos las que no existan (por si alguna falta en el CSV)
    selected_cols = [c for c in selected_cols if c in df.columns]

    # ------------------- Tensor final -------------------
    X = df[selected_cols]
    
    # Eliminar filas con NaN tanto en X como en etiquetas
    mask = ~X.isna().any(axis=1)
    X = X[mask]
    etiquetas = etiquetas[mask]

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    if X.isna().any().any():
        print("⚠️ Atención: Se encontraron NaNs en las estadísticas seleccionadas.")
        print(X.isna().sum())  # Muestra cuántos NaNs tiene cada columna
    else:
        print(f"✅ No se encontraron NaNs en X.Total de jugadores válidos: {X.shape[0]}")
        print(f"\n Datos de {csv_file} procesados correctamente (tipo_stats={tipo_stats}, n_columnas={X_tensor.shape[1]}). \n")
    return X_tensor, etiquetas


"""
print("PER100 COMPLETO")
X_completo_per100, y = procesar_datos(r"datasets\nba\finales\nba19_24_per100_entrenamiento.csv", tipo_stats="completo")
X_completo_per100, y = procesar_datos(r"datasets\nba\finales\scout_per100_nba24_25.csv", tipo_stats="completo")
X_completo_per100, y = procesar_datos(r"datasets\nba\finales\test_per100_nba18_19.csv", tipo_stats="completo")

print("PER36 COMPLETO")
X_completo36, y = procesar_datos(r"datasets\nba\finales\nba19_24_per36_entrenamiento.csv", tipo_stats="completo")
X_completo_per36, y = procesar_datos(r"datasets\nba\finales\scout_per36_nba24_25.csv", tipo_stats="completo")
X_completo_per36, y = procesar_datos(r"datasets\nba\finales\test_per36_nba18_19.csv", tipo_stats="completo")

print("PERGAME COMPLETO")
X_completoG, y = procesar_datos(r"datasets\nba\finales\nba19_24_pergame_entrenamiento.csv", tipo_stats="completo")
X_completo_pergame, y = procesar_datos(r"datasets\nba\finales\scout_pergame_nba24_25.csv", tipo_stats="completo")
X_completo_pergame, y = procesar_datos(r"datasets\nba\finales\test_pergame_nba18_19.csv", tipo_stats="completo")

"""

"""
print("PER100 VOL")
X_volumen100, y = procesar_datos(r"datasets\nba\finales\nba19_24_per100_entrenamiento.csv", tipo_stats="volumen")
X_completo_per100, y = procesar_datos(r"datasets\nba\finales\scout_per100_nba24_25.csv", tipo_stats="volumen")
X_completo_per100, y = procesar_datos(r"datasets\nba\finales\test_per100_nba18_19.csv", tipo_stats="volumen")


X_volumen36, y = procesar_datos(r"datasets\nba\finales\nba19_24_per36_entrenamiento.csv", tipo_stats="volumen")
X_completo_per36, y = procesar_datos(r"datasets\nba\finales\scout_per36_nba24_25.csv", tipo_stats="volumen")
X_completo_per36, y = procesar_datos(r"datasets\nba\finales\test_per36_nba18_19.csv", tipo_stats="volumen")


X_volumenG, y = procesar_datos(r"datasets\nba\finales\nba19_24_pergame_entrenamiento.csv", tipo_stats="volumen")
X_completo_pergame, y = procesar_datos(r"datasets\nba\finales\scout_pergame_nba24_25.csv", tipo_stats="volumen")
X_completo_pergame, y = procesar_datos(r"datasets\nba\finales\test_pergame_nba18_19.csv", tipo_stats="volumen")
"""


"""
print("PER100 TIRO")
X_tiro100, y = procesar_datos(r"datasets\nba\finales\nba19_24_per100_entrenamiento.csv", tipo_stats="tiro")
X_completo_per100, y = procesar_datos(r"datasets\nba\finales\scout_per100_nba24_25.csv", tipo_stats="tiro")
X_completo_per100, y = procesar_datos(r"datasets\nba\finales\test_per100_nba18_19.csv", tipo_stats="tiro")

print("PER36 TIRO")
X_tiro36, y = procesar_datos(r"datasets\nba\finales\nba19_24_per36_entrenamiento.csv", tipo_stats="tiro")
X_completo_per36, y = procesar_datos(r"datasets\nba\finales\scout_per36_nba24_25.csv", tipo_stats="tiro")
X_completo_per36, y = procesar_datos(r"datasets\nba\finales\test_per36_nba18_19.csv", tipo_stats="tiro")

print("PERGAME TIRO")
X_tiroG, y = procesar_datos(r"datasets\nba\finales\nba19_24_pergame_entrenamiento.csv", tipo_stats="tiro")
X_completo_pergame, y = procesar_datos(r"datasets\nba\finales\scout_pergame_nba24_25.csv", tipo_stats="tiro")
X_completo_pergame, y = procesar_datos(r"datasets\nba\finales\test_pergame_nba18_19.csv", tipo_stats="tiro")
"""