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
# class Document(models.Model):
#     docfile = models.FileField(upload_to='Re_entrenar/juanjofb/')

def user_directory_path(instance,filename):
    user = filename.split('-')
    # file will be uploaded to MEDIA_ROOT / user_<id>/<filename>
    return 'Re_entrenar/{0}/{1}'.format(user[0],filename)

class Document(models.Model):
    docfile = models.FileField(upload_to=user_directory_path)

########################################################################################################################