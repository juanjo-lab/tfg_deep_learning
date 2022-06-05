# -*- coding: utf-8 -*-

                                    #LIBRERIAS
########################################################################################################################
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from Predecir.models import *
import Predecir.models as models
from Predecir.forms import *
import Predecir.forms as forms
from django.shortcuts import render, redirect, get_object_or_404, render_to_response
from django.template import RequestContext
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statistics
from keras.models import model_from_json
########################################################################################################################
                                    #CODIGO PREDICCION
########################################################################################################################
def Predict(request,Patient):
    #  1/17
    print('\n Parte 1 de 17 - Importar Librerias \n')

    print('\n Completado sin errores las importaciones de librerias.  check 1/17')
    ########################################################################################################################

    ########################################################################################################################
    total_run_time_start_time = dt.datetime.now()
    print('Start time: ' + str(dt.datetime.now()))
    ########################################################################################################################


    # PREPARAR VARIABLES#
    ########################################################################################################################
    print('\n Preparar variables')

    initial_path = '/home/juanjo/PycharmProjects/glucosaRNN/Django/prueba_1/prueba_1/media/Predecir/Paciente_Nuevo/'
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
    #test_record_number = 657
    # Numero de reiteraciones neuronalessave
    num_epochs = 5
    # Tamaño del lot
    training_batch_size = 76
    # Establezca el valor de pérdida inicial en 0.
    # Esto es necesario para imprimir el gráfico si no se ejecuta el entrenamiento.
    loss = 0

    # PARA GUARDAR EL MODELO

    print('\n Completado sin errores las declaraciones iniciales de variables.')

    # model_file = input_train_file_name_prefix + '-' + str(num_epochs) + '.h5'
    # model_file_json = input_train_file_name_prefix + '-' + str(num_epochs) + '.json'
    model_file = input_train_file_name_prefix + '.h5'
    model_file_json = input_train_file_name_prefix  + '.json'
    # load weights into new model
    json_file = open(initial_path+model_file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)


    model.load_weights(initial_path + model_file)
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
    test_inputs = scaler.fit_transform(test_inputs)
    ## reajustar codigo
    test_features = []
    labels_test = []
    test_record_number = len(test_inputs)# horas de predicción
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
    plt.plot(predictions, color='blue', label='Valores de Glucosa predecidos')
    plt.axhline(y=180, color="red", linestyle = "--", linewidth = 5, label = "Limite Hiperglucemias")
    plt.axhline(y=150, color="black", linestyle = ":", linewidth = 3, label ="Ideal")
    plt.axhline(y=90, color="red", linestyle = "--", linewidth = 5, label = "Limite Hipoglucemias")
    plt.title('Valores de GLucosa predecidos LSTM')
    plt.xlabel('Puntos temporales')
    plt.ylabel('Valores de Glucosa [BG]')
    plt.legend()
    plt.savefig("/home/juanjo/PycharmProjects/glucosaRNN/Django/prueba_1/prueba_1/media/Predecir/Paciente_Nuevo/Prediccion_"+Patient+ ".jpg", bbox_inches="tight")
    plt.savefig("/home/juanjo/PycharmProjects/glucosaRNN/Django/prueba_1/prueba_1/usuario/static/media/Ultima_Prediccion/Prediccion_"+Patient+".png", bbox_inches="tight")

    ########################################################################################################################
########################################################################################################################

                                    #views
########################################################################################################################
def list(request):
    if request.user.is_anonymous():
        return render_to_response('usuario/not_user.html', context_instance=RequestContext(request))
    else:
        # Handle file upload
        if request.method == 'POST':
            form = forms.DocumentForm(request.POST, request.FILES)
            if form.is_valid():
                newdoc = models.Document(docfile=request.FILES['docfile'])
                newdoc.save()
                Predict(request,request.user.username)
                # Redirect to the document list after POST
                return HttpResponseRedirect(reverse('list_pre'))
        else:
            form = forms.DocumentForm()  # A empty, unbound form

        # Load documents for the list page
        documents = models.Document.objects.all()

        # Render list page with the documents and the form
        return render(request,'listo.html',{'documents': documents, 'form': form})
########################################################################################################################
