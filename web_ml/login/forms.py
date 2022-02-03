import imp
from django import forms
from django.contrib.auth.forms import UserCreationForm
from . import models

class SignupForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model   = models.CustomUser
        fields  = ("username","api_key","secret_key") #ここにパスワードを書くと被る