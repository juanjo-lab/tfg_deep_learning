# -*- coding: utf-8 -*-

from django.conf.urls import include, url
from Crear_red.views import *
import Crear_red.views as views

urlpatterns = [
    url(r'^Crear_red/$', views.list, name='list_cre')
]
