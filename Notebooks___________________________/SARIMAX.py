#-----------------------------------------------------------------------------------------------------------------------
# Creado 3/22
# Última actualizacion
# Nombre: Juanjo
#-----------------------------------------------------------------------------------------------------------------------
# env : conda activate pythonProject

# Se va a partir en varias ejecuciones el documento

#En principio, un modelo SARIMAX es un modelo de regresión lineal que utiliza un proceso de tipo SARIMA, es decir este
#modelo es útil en los casos donde sospechamos que los residuos pueden exhibir una tendencia estacional o patrón.




                                #IMPORTAR LIBRERIAS#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   1/6
print('\n Parte 1 de 6 - Importar Librerias \n')

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

total_run_time_start_time = dt.datetime.now() # Hora actual de comienzo
print('Start time: ' + str(total_run_time_start_time))

print('\n Completado sin errores las importaciones de librerias.  check 1/6')
########################################################################################################################











                                                #CARGAR DATOS DE PACIENTES#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   2/6
print('\n Parte 2 de 6 - Cargar datos de pacientes')

print(np.__version__)#ver version de numpy para LSTM se necesita la 1.18.1
print(pd.__version__)

#vamos a volcarnos los datos LIMPIOS del paciente que se va a entrenar

# Ubicación DE NUESTRA CARPETA
initial_path = '/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/data files/'
input_train_file_name_prefix = 'adolescent#100'
input_test_file_name_prefix = 'adolescent#100'

# Cargamos los datos procesados y limpios.
processed_train_file_name = input_train_file_name_prefix + '-train-processed.csv'
processed_test_file_name = input_test_file_name_prefix + '-test-processed.csv'


# Determinar si los datos deben ser mostrados
plot_data = 'Y' # Se evalua como Y o N : Y se muestra, N no se muestra

# Determinar si se debe ejecutar ARIMA paso a paso para encontrar p,d,q - el código está en la última celda
stepwise = 'Y'# Se evalua como Y o N : Y se muestra, N no se muestra


# Parametros de SARIMAX
pdq = (4, 0, 3) #(p,d,q)m
PDQm = (1, 1, 1, 7) #(P,D,Q,m)

print('\n Completado sin errores la preparacion de lectura de datos.  check 2/6')
########################################################################################################################









                                        #LEER DATOS DE PACIENTES#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   3/6

print('\n Parte 3 de 6 - Leer datos de paciente')

# Se leen los documentos
train_df = pd.read_csv(str(initial_path + processed_train_file_name), index_col=False)
test_df = pd.read_csv(str(initial_path + processed_test_file_name), index_col=False)

# Quitamos las columnas de fecha de nuestro csv en ambos
train_df.drop(['Time'], axis=1, inplace=True)
test_df.drop(['Time'], axis=1, inplace=True)

# Agregar ceros iniciales a Hora; de lo contrario, se muestra en el orden incorrecto
# 0, 1, 10, etc
train_df['Hour']=train_df['Hour'].apply(lambda x: '{0:0>2}'.format(x))
test_df['Hour']=test_df['Hour'].apply(lambda x: '{0:0>2}'.format(x))


# Combinar la fecha con la hora como un String
train_df['Date_Hour'] = train_df.Date.astype(str).str.cat(train_df.Hour.astype(str), sep=':')
test_df['Date_Hour'] = test_df.Date.astype(str).str.cat(test_df.Hour.astype(str), sep=':')

# ya no son necesarios, ni la fecha ni la hora
train_df.drop(['Date', 'Hour'], axis=1, inplace=True)
test_df.drop(['Date', 'Hour'], axis=1, inplace=True)

# Agrupamos los valores de BG con fechar_hora
train_grouped_df = train_df.groupby('Date_Hour', as_index=False)[['BG']].mean()
test_grouped_df = test_df.groupby('Date_Hour', as_index=False)[['BG']].mean()

print('\n Lectura correcta y ajustes correctos, check 3/6')
########################################################################################################################






                                        #PLOT TEST Y ENTRENAR DATOS#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   4/6

print('\n Parte 4 de 6 - Plot test y entrenar datos')

#Vamos a plotear datos
if(plot_data == 'Y'):
    # Trazar los datos de entrenamiento
    # Convierta la columna BG en una serie para que se pueda trazar
    train_grouped_series = train_grouped_df['BG']
    plt.figure(figsize=(20, 8))
    train_grouped_series.plot(kind='line',
                              title='Datos de entrenamiento para glucosa en sangre a lo largo del tiempo\nadolescent-2160-0-Dexcom-Cozmo-Basal\n' + input_train_file_name_prefix)

    plt.xlabel('Puntos en el tiempo')
    plt.ylabel('Niveles de Glucosa')
    plt.show()

    print('Valor inicial para datos de entrenamiento:')
    print(str(train_grouped_df.iloc[0, 0]) + ' ' + str(train_grouped_series[[0]]))
    print('\nNúmero de puntos de datos de entrenamiento: ' + str(len(train_grouped_series)))

    test_grouped_series = test_grouped_df['BG']
    plt.figure(figsize=(20, 8))
    test_grouped_series.plot(kind='line', color='green',
                             title='Datos de entrenamiento para glucosa en sangre a lo largo del tiempo\nadolescent-2160-0-Dexcom-Cozmo-Basal\n' +
                                   input_test_file_name_prefix)

    plt.xlabel('Time Point')
    plt.ylabel('Glucose Levels')
    plt.show()

    print('Valor inicial para datos de entrenamiento:')
    print(str(test_grouped_df.iloc[0, 0]) + ' ' + str(test_grouped_series[[0]]))
    print('\nNúmero de puntos de datos de entrenamiento: ' + str(len(test_grouped_series)))

print('\n Lectura correcta y ajustes correctos, check 4/6')
########################################################################################################################








                                    #DICKEY-FULLER#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   5/6
print('\n Parte 5 de 6 - prueba Dickey-Fuller')

# Ejecute la prueba Dickey-Fuller aumentada para verificar la estacionariedad

#La hipótesis nula es que los datos no son estacionarios, por lo que p < 0.05 indica estacionariedad
#y no necesitamos que el parámetro de diferenciación (d) cambie de 0 en el modelo ARIMA


result = adfuller(train_grouped_series)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))


print('\n Lectura correcta y ajustes correctos, check 5/6')
########################################################################################################################











                                            #ENTRENAMIENTO Y PREDICCION#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   6/6
print('\n Parte 6 de 6 - Entrenamiento y predicción')

#Cogemos el tiempo inicial de comienzo de entrenamiento y prediccion
train_predict_start_time = dt.datetime.now()

#concatenamos la secuencia de entramiento y test para trabajar unicamente con una
combined_series = pd.concat([train_grouped_series, test_grouped_series])

# Establecer sesiones de entrenamiento y prueba en series de entrada
train, test = combined_series[0:len(train_grouped_series)], combined_series[len(train_grouped_series):len(combined_series)]

#### train
# Definir modelo basado en datos de entrenamiento
train_model = sm.tsa.SARIMAX(combined_series[:4273], order=pdq, seasonal_order=PDQm)
# Fit
train_model_fit = train_model.fit()

### test
train_model = sm.tsa.SARIMAX(combined_series, order=pdq, seasonal_order=PDQm)
test_model_fit = train_model.filter(train_model_fit.params)

# correr las predicciones
predict_test = test_model_fit.get_prediction(start=4274, end=4931, dynamic=False, full_results=True)
predict_test = predict_test.predicted_mean

# Predictions se usa para mantener los valores predichos
predictions = []

# Para calcular el error, va comprobando valores
for i in range(4274,4931):
  predictions.append(predict_test.get(i))
  if ((i - 4274) % 20 == 0):
      print('Time point = ' + str(i) + ' - Predicted value = %.3f, Expected value = %.3f ' %
            (predict_test.get(i), test[i - 4274]))

#### CALCULAR EL MSE
mean_square = mean_squared_error(test, predictions)
print('\nTest MSE: %.3f' % mean_square)

# Obtenga la hora de finalización del entrenamiento y las predicciones para ver el tiempo que le lleva
train_predict_end_time = dt.datetime.now()

#ver el tiempo total
train_predict_total_time = (train_predict_end_time-train_predict_start_time).total_seconds()

print('\n\n Tiempo total de prediccion %.1f' % train_predict_total_time)
########################################################################################################################





############################################################################## Plot Predicciones

print(test)
plt.figure(figsize=(20,8))
plt.plot(test, color='blue', label='Actual Glucose Levels')#valores iniciales
plt.plot(predictions, color='red', label='Predicted Glucose Levels')#prediccion
plt.title('Niveles de GLucose predecidos - SARIMAX\nTraining File = adolescent-2160-0-Dexcom-Cozmo-Basal\n'
          + input_train_file_name_prefix
          + '\nTest File = adolescent-2160-0-Dexcom-Cozmo-Basal\n' + input_test_file_name_prefix +
          str('\n Error Cuadrático Medio = %.3f' % mean_square) +
         '\nParametros - (' + str(pdq) + ')(' + str(PDQm) + ')')
plt.xlabel('Puntos temporales')
plt.ylabel('Niveles de Glucosa')
plt.legend()
plt.show()


print('\n Plot + prediccion, check 6/6')
########################################################################################################################
