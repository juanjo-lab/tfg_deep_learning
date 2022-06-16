#!/usr/bin/env python
# -*- coding: utf-8 -*-
from django.conf.urls import url
from reentrenar.views import *
import reentrenar.views as views


urlpatterns = [
    url('RE_entrenar_1/',views.Re_entrenar,name = 're_entrenar_'),
]