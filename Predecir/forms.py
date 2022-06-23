# -*- coding: utf-8 -*-

from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='Seleccionar el archivo: ', help_text = 'username-test.csv'
    )
