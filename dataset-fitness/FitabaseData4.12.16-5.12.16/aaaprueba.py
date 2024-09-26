# PRUEBA

import pandas as pd

# Especifica la ruta del archivo CSV
archivo_csv = 'heartrate_seconds_merged.csv'

# Lee el archivo CSV y crea un DataFrame
dataframe = pd.read_csv(archivo_csv)

# Convert 'ActivityHour' a formato de fecha y hora
dataframe['Time'] = pd.to_datetime(dataframe['Time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Muestra el DataFrame
print(dataframe)

# Muestra la información del DataFrame
print(dataframe.info())

# Muestra las primeras filas del DataFrame
print(dataframe.head())
