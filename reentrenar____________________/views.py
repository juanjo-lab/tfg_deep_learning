#!/usr/bin/env python
# -*- coding: utf-8 -*-

from django.shortcuts import render, redirect, get_object_or_404, render_to_response
from usuario.views import *
from django.http import HttpResponse
from django.template import RequestContext


def Re_entrenar(request):
    if request.user.is_anonymous():
        return render_to_response('usuario/not_user.html', context_instance=RequestContext(request))
    else:
        if request.method == 'GET':
            print('Me están pidiendo un archivo')
            return render(request,'re_entrenar.html')
        elif request.method == 'POST':
            print('Me están enviando datos')
            print(request.body)
            return HttpResponse('Datos recibidos')
    return(request)
# Create your views here.
