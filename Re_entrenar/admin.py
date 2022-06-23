'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django.contrib import admin
from Re_entrenar.models import Document
########################################################################################################################

########################################################################################################################
admin.site.register(Document)
# Register your models here.
########################################################################################################################