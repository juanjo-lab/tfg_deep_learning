# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django.db import models
########################################################################################################################

########################################################################################################################
class Document(models.Model):
    docfile = models.FileField(upload_to='Crear_Red/juanjofb/')
########################################################################################################################
