#!/usr/bin/env python
# -*- coding: utf-8 -*-
from django import forms
from django import *
from django.contrib.auth.models import User, UserManager, Permission  # fill in custom user info then save it 
from django.contrib.auth.forms import UserCreationForm



class MyRegistrationForm(UserCreationForm):
    username = forms.CharField(required = True)
    

    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', 'first_name', 'last_name', 'email', 'groups', 'is_staff','is_active')
