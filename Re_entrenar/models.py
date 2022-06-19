# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django.db import models
########################################################################################################################

'''
Subida de elementos
'''
########################################################################################################################
class Document(models.Model):
    docfile = models.FileField(upload_to='Re_entrenar/juanjofb/')
########################################################################################################################