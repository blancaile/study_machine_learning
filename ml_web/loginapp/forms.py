from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from . import models

class CustomUserCreationForm(UserCreationForm):
    class Meta:
        model = models.loginmodel
        fields = ("username","password","api_key","secret_key")