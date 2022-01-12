from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import loginmodel  #読み込み
admin.site.register(loginmodel)