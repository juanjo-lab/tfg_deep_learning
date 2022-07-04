# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django.conf.urls import include, url
from Re_entrenar.views import *
import Re_entrenar.views as views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
########################################################################################################################

########################################################################################################################
urlpatterns = [
    url(r'^Re_entrenar/$', views.list, name='list_re')
]
urlpatterns += staticfiles_urlpatterns()
########################################################################################################################