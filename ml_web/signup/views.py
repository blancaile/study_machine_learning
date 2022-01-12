from django.shortcuts import render

from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views.generic import CreateView
# Create your views here.

class signup(CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy("login")#登録成功時に遷移するurl
    template_name = "signup.html"
