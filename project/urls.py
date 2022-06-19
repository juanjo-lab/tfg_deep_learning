#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django.conf.urls import include, url
from django.contrib import admin
########################################################################################################################

'''
url principales
'''
########################################################################################################################
urlpatterns = [
	url(r'^', include('usuario.urls')),
    url(r'^admin/', admin.site.urls),
    url(r'index/Crear_Red/', include('Crear_red.urls')),
    url(r'index/Re_entrenar/', include('Re_entrenar.urls')),
    url(r'index/Predecir/', include('Predecir.urls')),
]
########################################################################################################################