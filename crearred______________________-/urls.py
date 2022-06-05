#!/usr/bin/env python
# -*- coding: utf-8 -*-
from django.conf.urls import url

from crearred.views import *
import crearred.views as views


urlpatterns = [
    url('crearred_1/$',views.Crear_red,name = 'crearred_'),
#    url(r'upload/$', views.Upload)
]