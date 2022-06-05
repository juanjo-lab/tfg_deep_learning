#!/usr/bin/env python
# -*- coding: utf-8 -*-
from django.conf.urls import url
from predecir.views import *
import predecir.views as views

urlpatterns = [
    url('predecir_1/',views.Predecir,name = 'predecir_'),
]