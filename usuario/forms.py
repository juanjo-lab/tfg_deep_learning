#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django import forms
from django import *
from django.contrib.auth.models import User, UserManager, Permission  # fill in custom user info then save it 
from django.contrib.auth.forms import UserCreationForm
########################################################################################################################

'''
CLase de registro del modulo usuario.
'''
########################################################################################################################
class MyRegistrationForm(UserCreationForm):
    username = forms.CharField(required = True)
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', 'first_name', 'last_name', 'email', 'groups', 'is_staff','is_active')
########################################################################################################################