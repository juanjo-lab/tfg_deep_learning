# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección se requieren modificaciones para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
#librerias Django
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.shortcuts import render, redirect, get_object_or_404, render_to_response
from django.template import RequestContext
from django.core.mail import EmailMultiAlternatives
from django.template.loader import get_template
from django.conf import settings
#de python
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statistics
# DE CODIGO
from keras.models import model_from_json
from Predecir.models import *
import Predecir.models as models
from Predecir.forms import *
import Predecir.forms as forms
from email.mime.base import MIMEBase
from email import encoders
import os
########################################################################################################################
'''
Funcion principal de la seccion Prediccion
'''
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

                CF = request.POST['name_CF']

                print(CF)

                BOLO_12h,BOLO_24h,CF =Predict(request,request.user.username,CF)

                send_email(request.user.email,request.user.username,BOLO_12h,BOLO_24h,CF)

                # Redirect to the document list after POST
                return HttpResponseRedirect(reverse('index'))
        else:
            form = forms.DocumentForm()  # A empty, unbound form

        # Load documents for the list page
        documents = models.Document.objects.all()

        # Render list page with the documents and the form
        return render(request,'listo.html',{'documents': documents, 'form': form})
########################################################################################################################

'''
Funcion de mensajería
deberá modificar el directorio en filename_12h, filename_24h, filename_1mes: 
'''
########################################################################################################################
def send_email(mail,username,BOLO_12h,BOLO_24h,CF):
    context = ({'username': username,'BOLO_12h':BOLO_12h,'BOLO_24h':BOLO_24h,'CF':CF})
    template = get_template('correo.html')
    content = template.render(context)
    email = EmailMultiAlternatives(
        'Informe de Glucosa',
        'Prueba de python',
        settings.EMAIL_HOST_USER,
        [mail],
    )#(titulo de correo, mensaje descriptivo, cuenta de correo, destinatarios)

    #adjuntar archivos
    filename_12h = str(os.environ['DIR_PER']) + "/Predecir/" + username + "/Prediccion_" + username + "_12h.jpg"
    attachment_12h = open(filename_12h, 'rb')
    part_12h = MIMEBase('application', 'octet-stream')
    part_12h.set_payload((attachment_12h).read())
    encoders.encode_base64(part_12h)
    part_12h.add_header('Content-Disposition', "attachment; filename= " + "Prediccion_12h.jpg")

    filename_24h = str(os.environ['DIR_PER']) + "/Predecir/" + username + "/Prediccion_" + username + "_24h.jpg"
    attachment_24h = open(filename_24h, 'rb')
    part_24h = MIMEBase('application', 'octet-stream')
    part_24h.set_payload((attachment_24h).read())
    encoders.encode_base64(part_24h)
    part_24h.add_header('Content-Disposition', "attachment; filename= " + "Prediccion_24h.jpg")


    filename_1mes = str(os.environ['DIR_PER']) + "/Predecir/" + username + "/Prediccion_" + username + "_1mes.jpg"
    attachment_1mes = open(filename_1mes, 'rb')
    part_1mes = MIMEBase('application', 'octet-stream')
    part_1mes.set_payload((attachment_1mes).read())
    encoders.encode_base64(part_1mes)
    part_1mes.add_header('Content-Disposition', "attachment; filename= " + "Prediccion_1mes.jpg")

    #montaje de correo
    email.attach_alternative(content, 'text/html')
    email.attach(part_12h)
    email.attach(part_24h)
    email.attach(part_1mes)
    email.send()
########################################################################################################################

'''
Funcion de prediccion.
deberá modificar el directorio en initial_path, plt.savefig: 
'''
########################################################################################################################
def Predict(request,Patient,unidades):
    #  1/17
    print('\n Parte 1 de 17 - Importar Librerias \n')

    scaler = MinMaxScaler(feature_range=(0, 1))
    print('\n Completado sin errores las importaciones de librerias.  check 1/17')
    ########################################################################################################################

    ########################################################################################################################
    total_run_time_start_time = dt.datetime.now()
    print('Start time: ' + str(dt.datetime.now()))
    ########################################################################################################################

    # PREPARAR VARIABLES#
    ########################################################################################################################
    print('\n Preparar variables')

    initial_path = str(os.environ['DIR_PER'])+"/Predecir/" +Patient+'/'
    input_train_file_name_prefix = Patient
    input_test_file_name_prefix = Patient

    # Para coger los datos procesados o no
    processed_train_file_name = input_train_file_name_prefix + '-train.csv'
    processed_test_file_name = input_test_file_name_prefix + '-test.csv'

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
    test_df = pd.read_csv(str(initial_path + processed_test_file_name), index_col=False)
    train_df = pd.read_csv(str(str(os.environ['DIR_PER']) + '/Re_entrenar/' + Patient + '/' + processed_train_file_name), index_col=False)

    ########################################################################################################################

    # AGRUPAR DATOS#
    ########################################################################################################################
    #### test

    train_means = train_df['BG'].groupby([train_df['Date'], train_df['Hour']]).mean()
    test_means = test_df['BG'].groupby([test_df['Date'], test_df['Hour']]).mean()
    print('Ha finalizado la agrupación correctamente por fecha para test')
    print('\n Completado sin errores el tratamiento de los datos.')
    ########################################################################################################################

    # AÑADIR ELEMENTOS A TRAIN Y TEST#
    ########################################################################################################################
    print('\n Añadir elementos a test_glucose_df  \n')
    train_glucose_df = pd.DataFrame(columns=['Date_Hour', 'Glucose_Level'])
    for i in range(0, len(train_means)):
        temp_date = train_means.index[[i][0]]
        temp_date_hour = str(temp_date[0]) + ':' + str(temp_date[1])
        train_glucose_level = train_means[[i][0]]
        train_glucose_df.loc[len(train_glucose_df)] = [temp_date_hour, train_glucose_level]


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

########## test
    glucose_total = pd.DataFrame(columns=['Glucose_Level'])
    glucose_total = pd.concat((train_glucose_df['Glucose_Level'], test_glucose_df['Glucose_Level']), axis=0)
    test_inputs = glucose_total[len(train_glucose_df) - 60:].values
    test_inputs = test_inputs.reshape(-1, 1)
    Test_inputs = scaler.fit_transform(test_inputs)
    test_2 = scaler.fit(test_inputs)
    test_inputs = test_2.inverse_transform(Test_inputs)
    test_inputs = scaler.transform(test_inputs)

    ## reajustar codigo
    test_features = []
    labels_test = []
    test_record_number = len(test_inputs)# horas de predicción
    for i in range(60, test_record_number):
        test_features.append(test_inputs[i - 60:i, 0])
        labels_test.append(test_inputs[i, 0])

    test_features = np.array(test_features)
    test_features = np.reshape(test_features, [test_features.shape[0], test_features.shape[1], 1])
    ########################################################################################################################



    # PREDICCIÓN#
    ########################################################################################################################
    print('\n LSTM - Prediccion  \n')

    # Tiempo de comienzo
    inf_start_time = dt.datetime.now()
    # cargar modelo
    print("esto es lo que predice :")
    print(test_features)
    predictions = model.predict(test_features)

    # Prediccion de transformacion inversa
    predictions = scaler.inverse_transform(predictions)


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
    print(predictions)
    dates = test_glucose_df['Date_Hour']
    ##################################### 1 mes

    plt.figure(figsize=(20, 12))
    plt.plot(dates, predictions, color='blue', label='Valores de Glucosa predecidos')
    plt.axvline(x=hiperglucemia(predictions),linestyle="dashdot", color='red', linewidth = 3, label = "Mayor valor de glucosa registrado")
    plt.axvline(x=hipoglucemia(predictions),linestyle="dashdot",color='green', linewidth = 3, label = "Menor valor de glucosa registrado")
    plt.axhline(y=140, color="black", linestyle = ":", linewidth = 3, label ="Ideal")
    plt.axhspan(90,180,color="black",alpha=0.1,label="Rango Objetivo")
    plt.xticks(rotation=90)

    plt.title('Valores de GLucosa predecidos LSTM   1 Mes')
    plt.xlabel('Puntos temporales')
    plt.ylabel('Valores de Glucosa [BG]')
    plt.legend()
    plt.savefig( str(os.environ['DIR_PER']) + "/Predecir/" + Patient + "/Prediccion_" + Patient + "_1mes.jpg", bbox_inches="tight")
    ##########################################

    ########################################## 24 h

    CF, BOLO_24h = correccion(unidades, hiperglucemia(predictions[1:24]))

    plt.figure(figsize=(20, 12))
    plt.plot(dates[1:24], predictions[1:24], color='blue', label='Valores de Glucosa predecidos')
    plt.axvline(x=hiperglucemia(predictions[1:24]),linestyle="dashdot", color='red', linewidth = 3, label = "Mayor valor de glucosa registrado")
    plt.axvline(x=hipoglucemia(predictions[1:24]),linestyle="dashdot",color='green', linewidth = 3, label = "Menor valor de glucosa registrado")
    plt.axhline(y=140, color="black", linestyle = ":", linewidth = 3, label ="Ideal")
    plt.axhspan(90,180,color="black",alpha=0.1,label="rango objetivo")
    plt.xticks(rotation=90)
    plt.title('Valores de GLucosa predecidos LSTM      24 HORAS')
    plt.xlabel('Puntos temporales')
    plt.ylabel('Valores de Glucosa [BG]')
    plt.legend()
    plt.savefig( str(os.environ['DIR_PER']) + "/Predecir/" + Patient + "/Prediccion_" + Patient + "_24h.jpg",bbox_inches="tight")
    ####################################

    ################################# 12h
    CF, BOLO_12h = correccion(unidades, hiperglucemia(predictions[1:12]))

    plt.figure(figsize=(20, 12))
    plt.plot(dates[1:12],predictions[1:12], color='#0114FA', label='Valores de Glucosa predecidos')
    plt.axvline(x=hiperglucemia(predictions[1:12]),linestyle="dashdot", color='red', linewidth = 3, label = "Mayor valor de glucosa registrado")
    plt.axvline(x=hipoglucemia(predictions[1:12]),linestyle="dashdot",color='green', linewidth = 3, label = "Menor valor de glucosa registrado")
    plt.axhline(y=140, color="black", linestyle = ":", linewidth = 3, label ="Ideal")
    plt.axhspan(90,180,color="black",alpha=0.1,label="rango objetivo")
    plt.xticks(rotation=90)
    plt.title('Valores de GLucosa predecidos LSTM       12 HORAS')
    plt.xlabel('Puntos temporales')
    plt.ylabel('Valores de Glucosa [BG]')
    plt.legend()
    plt.savefig( str(os.environ['DIR_PER']) + "/Predecir/" + Patient + "/Prediccion_" + Patient + "_12h.jpg",bbox_inches="tight")

    return BOLO_12h,BOLO_24h,CF
    ########################################################################################################################
########################################################################################################################

'''
Funcion de busqueda de hiper e hipoglucemias.
'''
#########################################################################################################################
def hiperglucemia(predictions):
    position = np.where(predictions == np.max(predictions))
    return position[0][0]
def hipoglucemia(predictions):
    position = np.where(predictions == np.min(predictions))
    return position[0][0]
########################################################################################################################

'''
Funcion de busqueda de calculo de correccion de glucemia.
'''
########################################################################################################################
def correccion(unidades,valormax):
    CF =1700/int(unidades)
    if valormax > 180:
        bolo = round((valormax-180)/CF)
    else:
        bolo=0
    return round(CF),bolo
########################################################################################################################


'''
Funcion upload_files.
'''
########################################################################################################################
def upload_file(request):
    return str('Predecir/'+request.user.username+'/')
########################################################################################################################
