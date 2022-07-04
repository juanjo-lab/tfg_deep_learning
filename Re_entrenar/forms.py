# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django import forms
########################################################################################################################

########################################################################################################################
class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Seleccionar un archivo:', help_text='username-[train/test].csv'
    )
########################################################################################################################