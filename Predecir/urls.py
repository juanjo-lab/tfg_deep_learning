# -*- coding: utf-8 -*-

from django.conf.urls import include, url
from Predecir.views import *
import Predecir.views as views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    url(r'^Predecir/$', views.list, name='list_pre')
]

urlpatterns += staticfiles_urlpatterns()