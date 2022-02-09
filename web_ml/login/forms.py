import imp
from click import password_option
# from django import forms
# from django.contrib.auth.forms import UserCreationForm
# from . import models

# class SignupForm(UserCreationForm):
#     class Meta(UserCreationForm.Meta):
#         model   = models.CustomUser
#         fields  = ("username","password","api_key","secret_key") #ここにパスワードを書くと被る
from django.contrib.auth.forms import UserCreationForm
from django import forms
from . import models

class SignupForm(forms.Form):
    username = forms.CharField(label="username")
    password = forms.CharField(label="password",widget=forms.PasswordInput())
    api_key = forms.CharField(label="api_key")
    secret_key = forms.CharField(label="secret_key")

class ApplyForm(forms.Form):
    password = forms.CharField(label="password",widget=forms.PasswordInput())