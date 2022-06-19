#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django.conf.urls import url,include
from usuario.views import *
import usuario.views as views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
########################################################################################################################

'''
url principales
'''
########################################################################################################################
urlpatterns = [
	url(r'^$', views.Login, name='login'),
	url(r'^index/$', views.Index, name='index'),
	url(r'^logout$', views.Logout, name='logout'),
	url(r'^create_user$', views.Create_User, name='create_user'),
	url(r'^user_success$', views.User_Success, name='user_success'),
	url(r'^profile$', views.Profile, name='profile'),
	url(r'^change_password$',views.Change_Password, name='change_password'),
]
urlpatterns += staticfiles_urlpatterns()
########################################################################################################################