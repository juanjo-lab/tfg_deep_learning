# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección debera modificar el directorio:
'''
########################################################################################################################
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from Re_entrenar.models import *
import Re_entrenar.models as models
from Re_entrenar.forms import *
import Re_entrenar.forms as forms
from django.shortcuts import render, redirect, get_object_or_404, render_to_response
from django.template import RequestContext
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statistics
from keras.callbacks import EarlyStopping  # se usa si train_YN=Y
from keras.models import model_from_json
from keras.optimizers import Adam
import glob
from django.template.loader import get_template
from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from email.mime.base import MIMEBase
from email import encoders
########################################################################################################################

'''
Funcion principal de la seccion re_entrenar
deberá modificar el directorio en Numero: 
'''
########################################################################################################################
def list(request):
    if request.user.is_anonymous(): # si no tiene acceso el usuario, lo bloqueara
        return render_to_response('usuario/not_user.html', context_instance=RequestContext(request))
    else:
        # Handle file upload
        if request.method == 'POST':
            form = forms.DocumentForm(request.POST, request.FILES)
            if form.is_valid():
                newdoc = models.Document(docfile=request.FILES['docfile'])
                newdoc.save()
                Numero = len(glob.glob("/home/juanjo/PycharmProjects/glucosaRNN/Django/prueba_1/prueba_1/media/Crear_Red/" +str(request.user.username) + "/*.csv"))
                print("El numero de documentos .csv = " + str(Numero))

                if Numero >= 2:
                    LSTM_Re_entrenar(request,str(request.user.username))
                    send__email(request.user.email, request.user.username)
                # Redirect to the document list after POST
                return HttpResponseRedirect(reverse('list_re'))
        else:
            form = forms.DocumentForm()  # A empty, unbound form

        # Load documents for the list page
        documents = models.Document.objects.all()

        # Render list page with the documents and the form
        return render(request,'lista.html',{'documents': documents, 'form': form})
########################################################################################################################

'''
Funcion de mensajería
deberá modificar el directorio en filename: 
'''
########################################################################################################################
def send__email(mail,username):
    context = {'username': username}
    template = get_template('corre_o.html')
    content = template.render(context)
    email = EmailMultiAlternatives(
        'Informe de Glucosa',
        'Prueba de python',
        settings.EMAIL_HOST_USER,
        [mail],
    )#(titulo de correo, mensaje descriptivo, cuenta de correo, destinatarios)

    #adjuntar archivos
    filename ="<<PATH_USUARIO>>/prueba_1/media/Re_entrenar/"+username+"/Re_entrenar_"+username+".jpg"
    attachment = open(filename, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= " + "Reajuste.jpg")

    #montaje de correo
    email.attach_alternative(content, 'text/html')
    email.attach(part)
    #email.attach_file(logo_data(username))
    email.send()
########################################################################################################################

'''
Funcion de reajuste
deberá modificar el directorio en initial_path, initial_path_cp y en los plt.savefig : 
'''
########################################################################################################################
def LSTM_Re_entrenar(request,Patient):
# -----------------------------------------------------------------------------------------------------------------------
    #  1/17
    print('\n Parte 1 de 17 - Importar Librerias \n')
    print('\n Completado sin errores las importaciones de librerias.  check 1/17')
    ########################################################################################################################

    ########################################################################################################################
    total_run_time_start_time = dt.datetime.now()
    print('Start time: ' + str(dt.datetime.now()))
    ########################################################################################################################




                                            #PREPARAR VARIABLES#
    ########################################################################################################################
    print('\n Preparar variables')

    initial_path = '<<PATH_USUARIO>>/prueba_1/media/Re_entrenar/'+Patient+'/'
    initial_path_cp = '<<PATH_USUARIO>>/prueba_1/media/Predecir/'+Patient+'/'
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
    #train_record_number = 4274
    #test_record_number = 657
    # Numero de reiteraciones neuronales
    num_epochs = 5
    # Tamaño del lot
    training_batch_size = 76
    # Establezca el valor de pérdida inicial en 0.
    # Esto es necesario para imprimir el gráfico si no se ejecuta el entrenamiento.
    loss = 0

    # PARA GUARDAR EL MODELO

    print('\n Completado sin errores las declaraciones iniciales de variables.')

    model_file = input_train_file_name_prefix + '.h5'
    model_file_json = input_train_file_name_prefix +'.json'


    json_file = open(initial_path + model_file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(initial_path + model_file)
    json_file.close()

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
    train_record_number = len(glucose_training_set)
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
    test_record_number = len(test_inputs) #horas de test
    for i in range(60, test_record_number):
        test_features.append(test_inputs[i - 60:i, 0])
        labels_test.append(test_inputs[i, 0])
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    ########################################################################################################################



    # Reentrenar
    ###############################################################################################################################

    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
    # el learning_rate para el modelo real optimo es 0.0021651469485534417
    # el learning_rate para el modelo hibrido optimo es 0.010447687780690022
    opt = Adam(learning_rate=0.0021651469485534417, beta_1=0.9)
    model.compile(optimizer=opt, loss='mean_squared_error')

    # model.fit REENTRENAR LOS PESOS
    history = model.fit(features_set, labels, epochs=num_epochs, batch_size=training_batch_size, callbacks=[es])
    plt.figure(figsize=(20, 12))
    plt.figure()
    plt.xlabel("a Epoca")
    plt.ylabel("Magnitud de pérdida")
    plt.plot(history.history["loss"])
    plt.savefig("<<PATH_USUARIO>>/prueba_1/media/Re_entrenar/"+Patient+"/Perdidas_" + Patient + ".jpg",bbox_inches="tight")

###############################################################################################################################

    #GUARDAR
    ###############################################################################################################################

    model_json = model.to_json()
    with open(initial_path+model_file_json, "w") as json_file:
         json_file.write(model_json)
    with open(initial_path_cp + model_file, "w") as json_file:
         json_file.write(model_json)

          # serialize weights to HDF5
    model.save_weights(str(initial_path + model_file))
    model.save_weights(str(initial_path_cp + model_file))
    print("Saved model to disk [pesos]")
    ###############################################################################################################################


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

    print('\n\nPlot Prediccion')
    plt.figure(figsize=(20, 12))
    plt.plot(actual, color='blue', label='Valores de Glucosa Actuales')
    plt.plot(predictions, color='red', label='Valores de Glucosa predecidos')
    plt.title('Valores de GLucosa predecidos LSTM\nArchivo de entrenamiento = \n' +'\nPerdidas de entrenamiento= ' +str(loss) + '\nError Cuadratico Medio(MSE) = ' + str(mean_square) + '\n' + str(num_epochs) + ' Iteraciones, ' + 'Batch Size = ' + str(training_batch_size))
    plt.xlabel('Puntos temporales')
    plt.ylabel('Valores de Glucosa [BG]')
    plt.legend()
    plt.savefig("<<PATH_USUARIO>>/prueba_1/media/Re_entrenar/" + Patient + "/Re_entrenar_" + Patient + ".jpg",bbox_inches="tight")
    ########################################################################################################################


                                            #METRICAS#
    ########################################################################################################################
    # Datos adicionales, mejor mostrados
    total_run_time_end_time = dt.datetime.now()

    print('\n Datos adicionales  \n')
    from email.mime.base import MIMEBase
    from email import encoders
    print('Perdidas de entrenamiento = ' + str(loss))
    print('MSE = ' + str(mean_square))

    print('\n\nTiempo total de inferencia' + inf_total_time)

    print('\n\nTiempo total entrenamiento ' + str((total_run_time_end_time - total_run_time_start_time).total_seconds()) + ' segundos.')

    ########################################################################################################################
########################################################################################################################
