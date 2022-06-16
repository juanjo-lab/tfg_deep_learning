#!/usr/bin/env python
# -*- coding: utf-8 -*-
from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
	url(r'^', include('usuario.urls')),
    url(r'^admin/', admin.site.urls),
    #url(r'index/predecir/', include('predecir.urls')),
    #url(r'index/crearred/', include('crearred.urls')),
    #url(r'index/RE_entrenar/', include('reentrenar.urls')),
    url(r'index/Crear_Red/', include('Crear_red.urls')),
    url(r'index/Re_entrenar/', include('Re_entrenar.urls')),
    url(r'index/Predecir/', include('Predecir.urls')),
]
