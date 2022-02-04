from django.contrib import admin
from . import models
from django.contrib.auth.models import Group
# Register your models here.

admin.site.register(models.CustomUser)
admin.site.unregister(Group)