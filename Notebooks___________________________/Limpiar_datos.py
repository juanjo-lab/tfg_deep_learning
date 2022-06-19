import numpy as np
import pandas as pd

# ubicación del archivo
initial_path = '/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/data files/MODELOS/DATOS_MEDICOS/'

# nombre del archivo
file_name = 'datos_medicos.csv'

# LEER CSV
########################################################################################################################
datos_df = pd.read_csv(str(initial_path + file_name), index_col=False)
########################################################################################################################

# Eliminar columnas no necesarias
########################################################################################################################
datos_df = datos_df.iloc[1:]
datos_df = datos_df.drop(['Dispositivo','Número de serial','Insulina del cambio de usuario (unidades)','Insulina de corrección (unidades)'
,'Comida e insulina (unidades)','Cuerpos cetónicos mmol/L','Notas','Insulina de acción larga (unidades)','Insulina de acción larga no numérica'
,'Carbohidratos (porciones)','Carbohidratos (gramos)','Alimento no numérico','Insulina de acción rápida (unidades)','Insulina de acción rápida no numérica',
'Tipo de registro',],axis = 1)
########################################################################################################################

# Agrupaciones por columnas
########################################################################################################################
datos_df['BG'] = datos_df['Historial de glucosa mg/dL'] + datos_df['Escaneo de glucosa mg/dL'] + datos_df['Tira reactiva para glucosa mg/dL']
datos_df = datos_df.drop(['Historial de glucosa mg/dL','Escaneo de glucosa mg/dL','Tira reactiva para glucosa mg/dL'])

datos_df['Time'] = datos_df['Sello de tiempo del dispositivo']
datos_df = datos_df.drop(['Sello de tiempo del dispositivo'])

print(datos_df)
########################################################################################################################