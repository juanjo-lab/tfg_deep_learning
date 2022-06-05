


def Predict():
    #  1/17
    print('\n Parte 1 de 17 - Importar Librerias \n')
    import datetime as dt
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import statistics
    from keras.models import model_from_json

    print('\n Completado sin errores las importaciones de librerias.  check 1/17')
    ########################################################################################################################

    ########################################################################################################################
    total_run_time_start_time = dt.datetime.now()
    print('Start time: ' + str(dt.datetime.now()))
    ########################################################################################################################


    # PREPARAR VARIABLES#
    ########################################################################################################################
    print('\n Preparar variables')
    Patient = 'juanjo'

    initial_path = '/prueba_1/media/Predecir/Paciente_Nuevo/'
    input_train_file_name_prefix = Patient
    input_test_file_name_prefix = Patient

    # Determinar si los datos se les realiza un plot o no


    # Para coger los datos procesados o no
    processed_train_file_name = input_train_file_name_prefix + '-train.csv'
    processed_test_file_name = input_test_file_name_prefix + '-test.csv'

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

    model_file = input_train_file_name_prefix + '-' + str(num_epochs) + '.h5'
    model_file_json = input_train_file_name_prefix + '-' + str(num_epochs) + '.json'
    json_file = open('Modelo_Hibrido_reentrenado_3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(initial_path + model_file_json)
    # load weights into new model
    model.load_weights(initial_path + model_file)
    json_file.close()

    print("Loaded model from disk")
    ########################################################################################################################

    # ABRIR Y LEER FICHEROS#
    ########################################################################################################################
    train_df = pd.read_csv(str(initial_path + processed_train_file_name), index_col=False)
    test_df = pd.read_csv(str(initial_path + processed_test_file_name), index_col=False)
    ########################################################################################################################


    # AGRUPAR DATOS#
    ########################################################################################################################
    ##### train
    train_means = train_df['BG'].groupby([train_df['Date'], train_df['Hour']]).mean()
    print('Ha finalizado la agrupación correctamente por fecha para entrenamiento')

    #### test
    test_means = test_df['BG'].groupby([test_df['Date'], test_df['Hour']]).mean()
    print('Ha finalizado la agrupación correctamente por fecha para test')
    print('\n Completado sin errores el tratamiento de los datos.')
    ########################################################################################################################


    # AÑADIR ELEMENTOS A TRAIN Y TEST#
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






    scaler = MinMaxScaler(feature_range=(0, 1))

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







    # PREDICCIÓN#
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








    # PLOT PREDICCION#
    ########################################################################################################################
    actual = test_glucose_df['Glucose_Level'].values
    actual = actual[0:len(test_glucose_df)]
    print('\n\nPlot Prediccion')
    plt.figure(figsize=(20, 12))
    plt.plot(predictions, color='red', label='Valores de Glucosa predecidos')
    plt.title(
        'Valores de GLucosa predecidos LSTM\nArchivo de entrenamiento = \n' + '\nPerdidas de entrenamiento= ' +
        str(loss) + '\nError Cuadratico Medio(MSE) = ' + str(mean_square) + '\n' + str(num_epochs) + ' Iteraciones, ' +
        'Batch Size = ' + str(training_batch_size))
    plt.xlabel('Puntos temporales')
    plt.ylabel('Valores de Glucosa [BG]')
    plt.legend()
    plt.show()
    plt.savefig("/media/Predecir/Paciente_Nuevo/Prediccion_Nuevo_Paciente_" + str(num_epochs) + "", bbox_inches="tight")
    ########################################################################################################################

