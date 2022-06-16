# -*- coding: utf-8 -*-

from django.conf.urls import include, url
from Re_entrenar.views import *
import Re_entrenar.views as views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    url(r'^Re_entrenar/$', views.list, name='list_re')
]
urlpatterns += staticfiles_urlpatterns()