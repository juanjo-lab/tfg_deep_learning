# -*- coding: utf-8 -*-
from django.db import models
from django.contrib.auth.models import User



# class Document(models.Model,User.User):
#     docfile = models.FileField(upload_to='Predecir/'++'/')


def user_directory_path(instance,filename):
    user = filename.split('-')
    # file will be uploaded to MEDIA_ROOT / user_<id>/<filename>
    return 'Predecir/{0}/{1}'.format(user[0],filename)

class Document(models.Model):
    docfile = models.FileField(upload_to=user_directory_path)
