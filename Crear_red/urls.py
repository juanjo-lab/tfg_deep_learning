# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django.conf.urls import include, url
from Crear_red.views import *
import Crear_red.views as views
########################################################################################################################

########################################################################################################################
urlpatterns = [
    url(r'^Crear_red/$', views.list, name='list_cre')
]
########################################################################################################################
