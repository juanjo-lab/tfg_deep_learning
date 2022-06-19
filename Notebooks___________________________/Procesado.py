#-----------------------------------------------------------------------------------------------------------------------
# Creado 3/22
# Última actualizacion
# Nombre: Juanjo
#-----------------------------------------------------------------------------------------------------------------------
# conda activate pythonProject
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   1/7
print('\n Parte 1 de 7 - Importar Librerias \n')

import datetime as dt
import pandas as pd

# Tiempo de comienzo
start_time = dt.datetime.now()
print('Start time: ' + str(dt.datetime.now()))

print('\n Completado sin errores las importaciones de librerias.  check 1/7')

#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   2/7
print('\n Parte 2 de 7 - Crear path \n')

# path del archivo
initial_path = '/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/data files/'

input_train_file_name_prefix = 'adolescent#100'
input_test_file_name_prefix = 'adolescent#100'
train_file_name = input_train_file_name_prefix + '-train.csv'
test_file_name = input_test_file_name_prefix + '-test.csv'
processed_train_file_name = input_train_file_name_prefix + '-train-processed.csv'
processed_test_file_name = input_test_file_name_prefix + '-test-processed.csv'

print('\n Completado sin errores las importaciones de librerias.  check 2/7')

#for a in range(1,10):
#    str_train="input_train_file_name_prefix_" + str(a) + "_"
#    str_train = 'adult#00' + str(a)

#    str_test = "input_test_file_name_prefix_" + str(a) + "_"
#    str_test = 'adult#00' + str(a)

#    str_train_name = str_train_name
#    str_train_name = str_train + '-train.csv'

#    str_test_name = str_test + '-train.csv'

#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   3/7
# Procesar los archivos. Esto incluye lo siguiente:
# 1. Abra los archivos de tren y prueba y léalos en marcos de datos
# 2. Convierta el campo de tiempo en ambos df de un objeto a una fecha y hora para que pueda manipularse
# 3. Suelta columnas innecesarias en ambos df
# 4. Divida el campo Hora en fecha y hora en ambos df. Estos se utilizarán para agrupar cuando el
# Se ejecutan LSTM o ARIMA.
# 5. Guarde los marcos de datos de tren y prueba en archivos csv



print('\nComenzar a procesar archivos\n\n**********\n\n')

# Abra los archivos csv y léalos en marcos de datos

print('\n Parte 3 de 7 - Abrir archivos y crear dataframes \n')

# Leer y entrenar train_df
train_df = pd.read_csv(str(initial_path + train_file_name), index_col=False)
print('Fichero de entrenamiento abierto')
print('train_df shape = ' + str(train_df.shape))

# Leer el fichero into df
test_df = pd.read_csv(str(initial_path + test_file_name), index_col=False)
print('Fichero test file')
print('test_df shape = ' + str(test_df.shape))

print('\n Completado sin errores en la apertura de ficheros.  check 3/7')

#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   4/7
# Convierta el campo Hora en ambos df de un objeto a una fecha y hora para que pueda
# ser manipulado

print('\n Parte 4 de 7 - Convertir Tiempo \n')

train_df['Time'] = pd.to_datetime(train_df['Time'])
test_df['Time'] = pd.to_datetime(test_df['Time'])

print('\n Completado sin errores en la conversión de tiempos.  check 4/7')



#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   5/7
# Eliminar columnas innecesarias
print('\n Parte 5 de 7 - Drop columnas \n')
train_df.drop(['CGM', 'CHO', 'insulin', 'LBGI', 'HBGI', 'Risk'], axis=1, inplace=True)
test_df.drop(['CGM', 'CHO', 'insulin', 'LBGI', 'HBGI', 'Risk'], axis=1, inplace=True)
print('\n Completado sin errores en la extracción de columnas innesarias.  check 5/7')


#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   6/7

# Divida el campo Hora en fecha y hora en ambos df. Estos se utilizarán para agrupar cuando el
# Se ejecutan LSTM o ARIMA.
print('\n Parte 6 de 7 - Divida el tiempo para entrenar y probar, luego guarde los archivos procesados \n')

print('Tiempo de division de train - Puede tardar unos minutos\n')
for i in range(0, len(train_df)):
    train_df['Date'] = pd.DatetimeIndex(train_df['Time']).date
    train_df['Hour'] = pd.DatetimeIndex(train_df['Time']).hour

print('Tiempo de division de test - Puede tardar unos minutos\n')
for i in range(0, len(test_df)):
    test_df['Date'] = pd.DatetimeIndex(test_df['Time']).date
    test_df['Hour'] = pd.DatetimeIndex(test_df['Time']).hour

print('\n Completado sin errores las divisiones necesarias de ambos documentos.  check 5/7')


#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   7/7

# Guardar los archivos
print('\n Parte 7 de 7 - Guardar los datos procesados en los archivos de extensión csv \n')
train_df.to_csv(str(initial_path + processed_train_file_name), index=False)
#'/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/data files/'+'-train-processed.csv'
test_df.to_csv(str(initial_path + processed_test_file_name), index=False)
#'/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/data files/'+'-test-processed.csv'
print('\n Completado sin errores el volcado de datos.  check 7/7')

print('\n Los archivos han sido procesados .')

#-----------------------------------------------------------------------------------------------------------------------

end_time = dt.datetime.now()
print('TIempo de terminación: ' + str(dt.datetime.now()))
print('\n\nScript completado en ' + str((end_time-start_time).total_seconds()) + ' segundos.')



