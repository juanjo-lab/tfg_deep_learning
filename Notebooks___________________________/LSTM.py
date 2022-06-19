
#-----------------------------------------------------------------------------------------------------------------------
# Creado 3/22
# Última actualizacion
# Nombre: Juanjo
#-----------------------------------------------------------------------------------------------------------------------
# location =
# env = conda activate glucosa_2

#La memoria a corto plazo es una arquitectura de red neuronal recurrente artificial utilizada en el campo del aprendizaje
#profundo. A diferencia de las redes neuronales de retroalimentación estándar, LSTM tiene conexiones de retroalimentación.





########################################################################################################################
#                                                                                                                   1/17
print('\n Parte 1 de 17 - Importar Librerias \n')

import datetime as dt
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statistics
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
########################################################################################################################




########################################################################################################################
total_run_time_start_time = dt.datetime.now()
print('Start time: ' + str(dt.datetime.now()))

# Specify the GPU to use
gpu_number = str(0)

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number

print('\n Completado sin errores las importaciones de librerias.  check 1/17')
########################################################################################################################


                                    #PREPARAR VARIABLES#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   2/17

print('\n Parte 2 de 17 - Preparar variables')

# Establezca train_YN en Y si es necesario realizar un entrenamiento real. Si solo se ejecuta contra el conjunto de prueba,
# luego establezca en N.
train_YN = 'N'

#Buscamos la carpeta donde tenemos nuestros datos preparados
initial_path = '/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/data files/'


input_train_file_name_prefix = 'adolescent#001'
input_test_file_name_prefix = input_train_file_name_prefix


# Determinar si los datos se les realiza un plot o no
plot_data = 'N'#Y o N
plot_Final_Prediction = 'Y' # Y o N


# Para coger los datos procesados o no
processed_train_file_name = input_train_file_name_prefix + '-train-processed.csv'
processed_test_file_name = input_test_file_name_prefix + '-test-processed.csv'


# Establecer el valor de cuántos registros usar según el tamaño del conjunto de entrenamiento
# Si el tamaño del conjunto de entrenamiento es de 2160 horas, use 1824 puntos alrededor del 84% de los datos
train_record_number = 1824

# Establecer el valor de cuántos registros usar según el tamaño del conjunto de prueba
# Si el tamaño del conjunto de entrenamiento es de 2160 horas, use 396
test_record_number = 396

# Numero de reiteraciones neuronales
num_epochs = 90
# Tamaño del lot
training_batch_size = 32
# Establezca el valor de pérdida inicial en 0.
# Esto es necesario para imprimir el gráfico si no se ejecuta el entrenamiento.
loss = 0

# FIcha del modelo
model_file = input_train_file_name_prefix + '-' + str(num_epochs) + '-' + gpu_number + '.h5'

print('\n Completado sin errores las declaraciones iniciales de variables.  check 2/17')
########################################################################################################################




                                            #ABRIR Y LEER ARCHIVOS#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   3/17
print('\n Parte 3 de 17 - Abrir y leer ficheros procesados \n')

train_df = pd.read_csv(str(initial_path + processed_train_file_name), index_col=False)
test_df = pd.read_csv(str(initial_path + processed_test_file_name), index_col=False)

print('\n Completado sin errores las lectura de los ficheros.  check 3/17')
########################################################################################################################





                                          #AGRUPAR POR DATOS#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   4/17
print('\n Parte 4 de 17 - Agrupar por datos \n')
# Agrupamos los niveles de BG por Fecha y Hora en ambos df.
# Lo convertimos en un objeto Serie.

##### train
train_means = train_df['BG'].groupby([train_df['Date'], train_df['Hour']]).mean()
print('Ha finalizado la agrupación correctamente por fecha para entrenamiento')
#
#AGRUPA POR HORAS por ejemplo, el dia 5 a las 17:00 hasta las 18:00 tengo 10 medidas, las uno y hago la media y de 18:00
# a 19:00 hago lo mismo
#
#### test
test_means = test_df['BG'].groupby([test_df['Date'], test_df['Hour']]).mean()
print('Ha finalizado la agrupación correctamente por fecha para test')

print('\n Completado sin errores el tratamiento de los datos.  check 4/17')
########################################################################################################################


                                      #PLOTEO DE TRAIN Y TEST#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   5/17
print('\n Parte 5 de 17 - Ploteo de train y test  \n')

#### train

if(plot_data == 'Y'):
  plt.figure(figsize=(20,12))
  train_means.plot(kind='line', title='Datos de entrenamiento para glucosa en sangre a lo largo del tiempo \n adolescent-2160-0-Dexcom-Cozmo-Basal \n' + input_train_file_name_prefix)
  plt.xlabel('Puntos temporales')
  plt.ylabel('Niveles de glucosa')
  plt.show()

  print('Fecha de inicio de train_means')
  print(train_means[[0]])
  print('Fecha de terminación de train_means')
  print(train_means[[len(train_means) - 1]])

#### test

if (plot_data == 'Y'):
  plt.figure(figsize=(20, 12))
  np.flip(test_means).plot(kind='line', color='green',title="Datos de entrenamiento para glucosa en sangre a lo largo del tiempo \nadolescent-2160-0-Dexcom-Cozmo-Basal\n" +input_test_file_name_prefix)
  plt.xlabel('Puntos temporales')
  plt.ylabel('Niveles de glucosa')
  plt.show()

  print('Fecha de inicio de test_means')
  print(test_means[[0]])
  print('Fecha de terminación de test_means')
  print(test_means[[len(test_means) - 1]])



print('\n Completado sin errores el plot train y plot test.  check 5/17')



########################################################################################################################







                                    #AÑADIR ELEMENTOS A TRAIN Y TEST#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   6/17
print('\n Parte 6 de 17 - Añadir elementos a train_glucose_df y test_glucose_df  \n')

print('Se agregan elementos a train_glucose_df')

train_glucose_df = pd.DataFrame(columns=['Date_Hour', 'Glucose_Level'])
for i in range(0, len(train_means)):
  temp_date = train_means.index[[i][0]]
  temp_date_hour = str(temp_date[0]) + ':' + str(temp_date[1])
  train_glucose_level = train_means[[i][0]]
  train_glucose_df.loc[len(train_glucose_df)] = [temp_date_hour, train_glucose_level]
print('Terminado - train_glucose_df')


print('Se agregan elementos a test_glucose_df')
test_glucose_df = pd.DataFrame(columns=['Date_Hour', 'Glucose_Level'])
for i in range(0, len(test_means)):
  temp_date = test_means.index[[i][0]]
  temp_date_hour = str(temp_date[0]) + ':' + str(temp_date[1])
  test_glucose_level = test_means[[i][0]]
  test_glucose_df.loc[len(test_glucose_df)] = [temp_date_hour, test_glucose_level]
print(test_glucose_df)
print('Terminado - test_glucose_df')

print('\n Completado sin errores el tratamiento de train y test.  check 6/17')
########################################################################################################################




                                            #CONJUNTO DE ENTRENAMIENTO DE ESCALA#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   7/17

print('\n Parte 7 de 17 - Comienzo de LSTM - CONJUNTO DE ENTRENAMIENTO DE ESCALA  \n')

glucose_training_set = train_glucose_df.iloc[:,1:2].values

# Escalado de características

scaler = MinMaxScaler(feature_range = (0,1))
glucose_training_set_scaled = scaler.fit_transform(glucose_training_set)

print('\n Completado sin errores el escalado para valores de entrada.  check 7/17 \n')
########################################################################################################################


                                        #PREPARAR CONJUNTO DE ENTRENAMIENTO#
########################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   8/17
print('\n Parte 8 de 17 - LSTM - Preparar conjunto de entrenamiento  \n')
features_set = []
labels = []
train_record_number = len(glucose_training_set)
for i in range(60, train_record_number):
  features_set.append(glucose_training_set_scaled[i - 60:i, 0])
  labels.append(glucose_training_set_scaled[i, 0])
print('\n Completado sin errores el preparado de conjuntos.  check 8/17 \n')
########################################################################################################################







                                          #RESHAPE#
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                                   9/17
print('\n Parte 9 de 17 - LSTM - Reshape  \n')
features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
print('\n Completado sin errores el Reshape.  check 9/19 \n')
########################################################################################################################















                                    #CONTRUIR EL MODELO#
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                                  10/17

if train_YN == 'Y':

  print('\n Parte 10 de 17 - LSTM - Construir el modelo LSTM (si train_YN==Y)  \n')
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
  model.add(Dropout(0.2))
  #Dropout proporciona aumentos de precision.
  #Es un algoritmo bastante simplem en cada paso de entrenamiento, cada neutona tiene una probabilidad p de ser
  #temporalmente"abandonada", lo que significa que se ignorará por completo durante ete entrenamiento , pero puede
  #estar activo durante el siguiente paso.
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=50))
  model.add(Dropout(0.2))
  model.add(Dense(units = 1))#mejor un softmax para que el resultado lo de en estadisticos
  #Si usamos Dense, tenemos que asegurarnos de tener suficientes datos para no sobreajustar
  from keras.optimizers import Adam
  opt = Adam(learning_rate=0.000001, beta_1=0.9) # mal [0.1-0.001]
  model.compile(optimizer = opt, loss = 'mean_squared_error')
  #Se usa adam porque es la combinación de Momentum optimization y RMSRop, puesto que realiza un seguimiento de un
  #promedio exponencialmente decreciente de gradientes pasados, y al igual que RMSProp, mantiene pista de un promedio
  #exponencialmente decreciente de gradiantes cuadrados pasados.
print('\n Completado sin errores la construccion del modelo LSTM.  check 10/17 \n')

########################################################################################################################






                                      #CREACION DEL MODELO#
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                                  11/17
# Creacion del modelo
# Se ejecuta si train_YN == Y
from keras.optimizers import Adam

if train_YN == 'Y':
  print('\n Parte 11 de 17 - LSTM - Creación del  modelo LSTM (si train_YN==Y)  \n')
  # Tiempo de comienzo
  training_start_time = dt.datetime.now()
  model.summary()
  # Parada del paciente
  es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=90)
  history = model.fit(features_set, labels, epochs=num_epochs, batch_size=training_batch_size, callbacks=[es])
  loss = model.evaluate(features_set, labels, verbose=0)
  print('\nPerdida sin escala = ' + str(loss) )
  #######
  # Guardar MODELO de ENTRENAMIENTO

  # # serialize model to JSON
  # model_json = model.to_json()
  # with open("Modelo_Hibrido.json", "w") as json_file:
  #   json_file.write(model_json)
  #
  # # serialize weights to HDF5
  # model.save_weights(str(initial_path + model_file))
  # print("Saved model to disk")

  plt.figure(figsize=(20, 12))
  plt.figure()
  plt.xlabel("a Epoca")
  plt.ylabel("Magnitud de pérdida")
  plt.plot(history.history["loss"])

  #model.save(str(initial_path + model_file))

  # # COGER TIEMPOS DE ENTRENAMIENTO
  training_end_time = dt.datetime.now()
  #
  # # Ver el tiempo total de entrenamiento
  training_total_time = str((training_end_time - training_start_time).total_seconds()) + ' seconds.'
  print('\n\n Tiempo total de entrenamiento ' + training_total_time)
print('\n Completado sin errores la creacion del modelo LSTM.  check 11/17 \n')
########################################################################################################################





















                                    #CONCATENACION DE TRAIN Y TEST#
########################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------
#                                                                                                                  12/17
print('\n Parte 12 de 17 - LSTM - Concatenacion de train y test  \n')
glucose_total = pd.DataFrame(columns=['Glucose_Level'])
glucose_total = pd.concat((train_glucose_df['Glucose_Level'], test_glucose_df['Glucose_Level']), axis=0)
print('\n Completado sin errores la concatenacion.  check 12/17 \n')
########################################################################################################################




                                        #RESHAPE#
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                                  13/17
print('\n Parte 13 de 17 - LSTM - Reshape test shape  \n')
test_inputs = glucose_total[len(train_glucose_df) - 60:].values
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)
print(test_inputs)
print('\n Completado sin errores el reshape test.  check 13/17 \n')
########################################################################################################################

                                    #PREPARAR TEST#
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                                  14/17
#Preparacion del conjunto de prueba test
print('\n Parte 14 de 17 - LSTM - Preparar test  \n')
test_features = []
test_record_number = len(test_inputs)
for i in range(60, test_record_number):
  test_features.append(test_inputs[i - 60:i, 0])
test_features = np.array(test_features)
print(test_features.shape[0])
test_features = np.reshape(test_features, [test_features.shape[0], test_features.shape[1], 1])
print(test_features)
print('\n Completado sin errores la preparacion del test.  check 14/17 \n')
########################################################################################################################




                                        #PREDICCION#
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                                  15/17
# Prediccion
print('\n Parte 15 de 17 - LSTM - Prediccion  \n')
# # # Tiempo de comienzo
inf_start_time = dt.datetime.now()
# # # load json and create model
json_file = open('/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/Notebooks/Prediccion_ficticia.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/Notebooks/Prediccion_ficticia.h5')
print("Loaded model from disk")
predictions = model.predict(test_features)
# Prediccion de transformacion inversa
predictions = scaler.inverse_transform(predictions)
actual_predicted_difference_list = []

for i in range(0,len(predictions)):
     print('\n Valor actual: ' + str(test_glucose_df.loc[i,'Date_Hour']) + ' = '
           + str(test_glucose_df.loc[i,'Glucose_Level']))
     print('Valor predecido = {0}'.format(predictions[i,0]))

    # Calcular el error cuadratico medio
actual_predicted_difference_list.append((predictions[i,0] - test_glucose_df.loc[i,'Glucose_Level']) ** 2)

mean_square = statistics.mean(actual_predicted_difference_list)
print('Error cuadratico medio= ' + str(mean_square))

# Tiempo final de ejecucion
inf_end_time = dt.datetime.now()

# Tiempo total de ejecucion
#inf_total_time = str((inf_end_time-inf_start_time).total_seconds()) + ' segundos.'

#print('\n\n Tiempo total implicado ' + inf_total_time)

print('\n Completado sin errores la prediccion del conjunto.  check 15/17 \n')
########################################################################################################################


                                        #PREDICION#
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                                  16/17
#Plot predictions
print('\n Parte 16 de 17 - LSTM - Plot Prediccion  \n')
actual = test_glucose_df['Glucose_Level'].values
actual = actual[0:len(test_glucose_df)]
dates = test_glucose_df['Date_Hour']
print(predictions.max())
position = np.where(predictions==np.max(predictions))

if plot_Final_Prediction == 'Y':
  print('\n\nPlot Prediccion')
  plt.figure(figsize=(20, 12))
  #plt.plot(actual, color='blue', label='Valores de Glucosa Actuales')
  plt.plot(dates,predictions, color='red', label='Valores de Glucosa predecidos')
  plt.axvline(x=position[0][0],color= 'green', label='Hiperglucemia')
  plt.title(
    'Valores de GLucosa predecidos LSTM\nArchivo de entrenamiento = adolescent-2160-0-Dexcom-Cozmo-Basal\n' + input_train_file_name_prefix
    + '\nArchivo de test= adolescent-2160-0-Dexcom-Cozmo-Basal\n' + input_test_file_name_prefix + '\nPerdidas de entrenamiento= ' +
    str(loss) + '\nError Cuadratico Medio(MSE) = ' + str(mean_square) + '\n' + str(num_epochs) + ' Iteraciones, ' +
    'Batch Size = ' + str(training_batch_size))
  plt.xlabel('Puntos temporales')
  plt.ylabel('Valores de Glucosa')
  plt.legend()
  plt.show()
print('\n Completado sin errores el plot de las predicciones.  check 16/17 \n')
########################################################################################################################





                                            #Inf extra#
########################################################################################################################
# ----------------------------------------------------------------------------------------------------------------------
#                                                                                                                  17/17
# Datos adicionales, mejor mostrados
total_run_time_end_time = dt.datetime.now()

# Tiempo total de ejecucion
inf_total_time = str((inf_end_time-inf_start_time).total_seconds()) + ' segundos.'
print('\n Parte 17 de 17 - LSTM - Datos adicionales  \n')

print('Perdidas de entrenamiento = ' + str(loss))
print('MSE = ' + str(mean_square))

print('\n\nTiempo total de inferencia' + inf_total_time)

print('\n\nTiempo total entrenamiento ' + str((total_run_time_end_time-total_run_time_start_time).total_seconds()) + ' segundos.')

print('\n Fin.  check 17/17 \n')
#######################################################################################################################


