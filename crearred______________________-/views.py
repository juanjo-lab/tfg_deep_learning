#!/usr/bin/env python
# -*- coding: utf-8 -*-

from django.shortcuts import render, redirect, get_object_or_404, render_to_response
from usuario.views import *
from django.http import HttpResponse
from django.template import RequestContext



def Crear_red(request):
    if request.user.is_anonymous():
        return render_to_response('usuario/not_user.html', context_instance=RequestContext(request))
    else:
        if request.method == 'GET':
            print('Me están pidiendo un archivo')
            return render(request,'crearred.html')
        elif request.method == 'POST':
            print('Me están enviando datos')
            print(request.body)
            return HttpResponse('Datos recibidos')
    return(request)
# Create your views here.
#
# def Upload(request):
#
#     for count, x in enumerate(request.FILES.getlist("files")):
#         def process(f):
#             with open("/home/juanjo/PycharmProjects/glucosaRNN/Django/prueba_1/prueba_1/media/file_"+str(count),'wb+') as destination:
#                 for chunk in f.chunks():
#                     destination.write(chunk)
#
#         return HttpResponse("File(s) uploaded !!")
#         process(x)
#     return render(request,'crearred.html')
#
