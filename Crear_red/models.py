# -*- coding: utf-8 -*-
'''
Librerias emplementadas para el modulo de ususario, en esta sección no se requiere ninguna modificacion para el usuario
que desee implementar la aplicación.
'''
########################################################################################################################
from django.db import models
########################################################################################################################

########################################################################################################################
# class Document(models.Model):
#     docfile = models.FileField(upload_to='Crear_Red/juanjofb/')

def user_directory_path(instance,filename):
    user = filename.split('-')
    # file will be uploaded to MEDIA_ROOT / user_<id>/<filename>
    return 'Crear_Red/{0}/{1}'.format(user[0],filename)

class Document(models.Model):
    docfile = models.FileField(upload_to=user_directory_path)

########################################################################################################################
