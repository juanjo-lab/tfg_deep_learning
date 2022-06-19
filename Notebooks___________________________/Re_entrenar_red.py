# -----------------------------------------------------------------------------------------------------------------------
# Creado 3/22
# Última actualizacion
# Nombre: Juanjo
# -----------------------------------------------------------------------------------------------------------------------
# location =
# env = conda activate glucosa_2

# La memoria a corto plazo es una arquitectura de red neuronal recurrente artificial utilizada en el campo del aprendizaje
# profundo. A diferencia de las redes neuronales de retroalimentación estándar, LSTM tiene conexiones de retroalimentación.

# -----------------------------------------------------------------------------------------------------------------------
#  1/17
print('\n Parte 1 de 17 - Importar Librerias \n')
import datetime as dt
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statistics
from keras.callbacks import EarlyStopping  # se usa si train_YN=Y
import tensorflow as tf
from tensorflow import keras
from keras.layers import Bidirectional
from keras.models import model_from_json
from keras.optimizers import Adam
print('\n Completado sin errores las importaciones de librerias.  check 1/17')
########################################################################################################################









########################################################################################################################
total_run_time_start_time = dt.datetime.now()
print('Start time: ' + str(dt.datetime.now()))

# Specify the GPU to use
gpu_number = str(0)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
########################################################################################################################







                                        #PREPARAR VARIABLES#
########################################################################################################################
print('\n Preparar variables')
Patient = 'adolescent#100'
#Patient_Model = 'adolescent#001'

transfer_learning = 'Y'

initial_path = '/home/juanjo/PycharmProjects/glucosaRNN/Glucose-Level-Prediction-master/data files/'
input_train_file_name_prefix = Patient
input_test_file_name_prefix = Patient

# Determinar si los datos se les realiza un plot o no
plot_data = 'N'  # Y o N
plot_Final_Prediction = 'Y'  # Y o N

# Para coger los datos procesados o no
processed_train_file_name = input_train_file_name_prefix + '-train-processed.csv'
processed_test_file_name = input_test_file_name_prefix + '-test-processed.csv'

# Establecer el valor de cuántos registros usar según el tamaño del conjunto de entrenamiento
# Si el tamaño del conjunto de entrenamiento es de 2160 horas, use 1824 puntos alrededor del 84% de los datos
# Establecer el valor de cuántos registros usar según el tamaño del conjunto de prueba
# Si el tamaño del conjunto de entrenamiento es de 2160 horas, use 396
train_record_number = 4274
test_record_number = 657
# Numero de reiteraciones neuronales
num_epochs = 5
# Tamaño del lot
training_batch_size = 76
# Establezca el valor de pérdida inicial en 0.
# Esto es necesario para imprimir el gráfico si no se ejecuta el entrenamiento.
loss = 0

# PARA GUARDAR EL MODELO

print('\n Completado sin errores las declaraciones iniciales de variables.')


model_file = 'Modelo_Hibrido_reentrenado_3.h5'
json_file = open('Modelo_Hibrido_reentrenado_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_file)
print("Loaded model from disk")
########################################################################################################################









                                    #ABRIR Y LEER FICHEROS#
########################################################################################################################
train_df = pd.read_csv(str(initial_path + processed_train_file_name), index_col=False)
test_df = pd.read_csv(str(initial_path + processed_test_file_name), index_col=False)
########################################################################################################################











                                    #AGRUPAR DATOS#
########################################################################################################################
##### train
train_means = train_df['BG'].groupby([train_df['Date'], train_df['Hour']]).mean()
print('Ha finalizado la agrupación correctamente por fecha para entrenamiento')

#### test
test_means = test_df['BG'].groupby([test_df['Date'], test_df['Hour']]).mean()
print('Ha finalizado la agrupación correctamente por fecha para test')
print('\n Completado sin errores el tratamiento de los datos.')
########################################################################################################################








                                    #PLOT DATOS#
########################################################################################################################
print('\n Ploteo de train y test  \n')

#### train

if (plot_data == 'Y'):
    plt.figure(figsize=(20, 12))
    train_means.plot(kind='line',
                     title='Datos de entrenamiento para glucosa en sangre a lo largo del tiempo \n' +
                           input_train_file_name_prefix)
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
    test_means.plot(kind='line', color='green',
                    title="Datos de entrenamiento para glucosa en sangre a lo largo del tiempo\n" +
                          input_test_file_name_prefix)
    plt.xlabel('Puntos temporales')
    plt.ylabel('Niveles de glucosa')
    plt.show()

    print('Fecha de inicio de test_means')
    print(test_means[[0]])
    print('Fecha de terminación de test_means')
    print(test_means[[len(test_means) - 1]])
print('\n Completado sin errores el plot train y plot test.')
########################################################################################################################












                                    #AÑADIR ELEMENTOS A TRAIN Y TEST#
########################################################################################################################
print('\n Añadir elementos a train_glucose_df y test_glucose_df  \n')
############### train
train_glucose_df = pd.DataFrame(columns=['Date_Hour', 'Glucose_Level'])
for i in range(0, len(train_means)):
    temp_date = train_means.index[[i][0]]
    temp_date_hour = str(temp_date[0]) + ':' + str(temp_date[1])
    train_glucose_level = train_means[[i][0]]
    train_glucose_df.loc[len(train_glucose_df)] = [temp_date_hour, train_glucose_level]
print('Terminado - train_glucose_df')
############# test
print('Se agregan elementos a test_glucose_df')
test_glucose_df = pd.DataFrame(columns=['Date_Hour', 'Glucose_Level'])
for i in range(0, len(test_means)):
    temp_date = test_means.index[[i][0]]
    temp_date_hour = str(temp_date[0]) + ':' + str(temp_date[1])
    test_glucose_level = test_means[[i][0]]
    test_glucose_df.loc[len(test_glucose_df)] = [temp_date_hour, test_glucose_level]
print('Terminado - test_glucose_df')
print('\n Completado sin errores el tratamiento de train y test.')
########################################################################################################################











                                #CONJUNTO DE ENTRENAMIENTO DE ESCALA#
########################################################################################################################
print('\n Comienzo de  realimentacion de LSTM - CONJUNTO DE ENTRENAMIENTO DE ESCALA  \n')
######### train
glucose_training_set = train_glucose_df.iloc[:, 1:2].values
#ajustar caracteristicas
scaler = MinMaxScaler(feature_range=(0, 1))
glucose_training_set_scaled = scaler.fit_transform(glucose_training_set)
#Preparar conjunto de entrenamiento
features_set = []
labels = []
for i in range(60, train_record_number):
    features_set.append(glucose_training_set_scaled[i - 60:i, 0])
    labels.append(glucose_training_set_scaled[i, 0])
#Reshape
features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

########## test
glucose_total = pd.DataFrame(columns=['Glucose_Level'])
glucose_total = pd.concat((train_glucose_df['Glucose_Level'], test_glucose_df['Glucose_Level']), axis=0)
test_inputs = glucose_total[len(train_glucose_df) - 60:].values
test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)
## reajustar codigo
test_features = []

labels_test = []
for i in range(60, test_record_number):
    test_features.append(test_inputs[i - 60:i, 0])
    labels_test.append(test_inputs[i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
########################################################################################################################










                                    #TRANSFER LEARNING#
########################################################################################################################
if transfer_learning == "Y":
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
    # el learning_rate para el modelo real optimo es 0.0021651469485534417
    # el learning_rate para el modelo hibrido optimo es 0.010447687780690022
    opt = Adam(learning_rate= 0.010447687780690022, beta_1 = 0.9)
    model.compile(optimizer = opt, loss = 'mean_squared_error')

        # model.fit REENTRENAR LOS PESOS
    history = model.fit(features_set, labels, epochs=num_epochs, batch_size=training_batch_size, callbacks=[es])
    plt.figure(figsize=(20, 12))
    plt.figure()
    plt.xlabel("a Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(history.history["loss"])
    ##

########################################################################################################################













                                    #GUARDAR MODELO#
########################################################################################################################
# .h5
#     model_json = model.to_json()
#     with open('Modelo_real_reentrenado_2.json', "w") as json_file:
#         json_file.write(model_json)
#     model.save("Modelo_real_reentrenado_2.h5")
#         # COGER TIEMPOS DE ENTRENAMIENTO
#     training_end_time = dt.datetime.now()


# Ver el tiempo total de entrenamientof
    # training_total_time = str((training_end_time - training_start_time).total_seconds()) + ' seconds.'
    # print('\n\n Tiempo total de entrenamiento ' + training_total_time)
########################################################################################################################










                                    #PREDICCIÓN#
########################################################################################################################
print('\n LSTM - Prediccion  \n')

# Tiempo de comienzo
inf_start_time = dt.datetime.now()
# cargar modelo

predictions = model.predict(test_features)

# Prediccion de transformacion inversa

predictions = scaler.inverse_transform(predictions)

actual_predicted_difference_list = []

for i in range(0, len(predictions)):
    print('\n Valor actual: ' + str(test_glucose_df.loc[i, 'Date_Hour']) + ' = '
          + str(test_glucose_df.loc[i, 'Glucose_Level']))
    print('Valor predecido = {0}'.format(predictions[i, 0]))

    # Calcular el error cuadratico medio
    actual_predicted_difference_list.append((predictions[i, 0] - test_glucose_df.loc[i, 'Glucose_Level']) ** 2)

mean_square = statistics.mean(actual_predicted_difference_list)
print('Error cuadratico medio= ' + str(mean_square))

# Tiempo final de ejecucion
inf_end_time = dt.datetime.now()

# Tiempo total de ejecucion
inf_total_time = str((inf_end_time - inf_start_time).total_seconds()) + ' segundos.'

print('\n\n Tiempo total implicado ' + inf_total_time)
########################################################################################################################












                                    #PLOT PREDICCION#
########################################################################################################################
actual = test_glucose_df['Glucose_Level'].values
actual = actual[0:len(test_glucose_df)]

if plot_Final_Prediction == 'Y':
    print('\n\nPlot Prediccion')
    plt.figure(figsize=(20, 12))
    plt.plot(actual, color='blue', label='Valores de Glucosa Actuales')
    plt.plot(predictions, color='red', label='Valores de Glucosa predecidos')
    plt.title(
        'Valores de GLucosa predecidos LSTM\nArchivo de entrenamiento = \n' +'\ntrain :\n'+ input_train_file_name_prefix
        + '\ntrain :\n' +
        input_test_file_name_prefix + '\nPerdidas de entrenamiento= ' +
        str(loss) + '\nError Cuadratico Medio(MSE) = ' + str(mean_square) + '\n' + str(num_epochs) + ' Iteraciones, ' +
        'Batch Size = ' + str(training_batch_size))
    plt.xlabel('Puntos temporales')
    plt.ylabel('Valores de Glucosa [BG]')
    plt.legend()
    plt.show()
########################################################################################################################








                                        #METRICAS#
########################################################################################################################
# Datos adicionales, mejor mostrados
total_run_time_end_time = dt.datetime.now()

print('\n Datos adicionales  \n')

print('Perdidas de entrenamiento = ' + str(loss))
print('MSE = ' + str(mean_square))

print('\n\nTiempo total de inferencia' + inf_total_time)

print('\n\nTiempo total entrenamiento ' + str(
    (total_run_time_end_time - total_run_time_start_time).total_seconds()) + ' segundos.')

########################################################################################################################
