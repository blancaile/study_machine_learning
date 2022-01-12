from django.http.response import HttpResponse
from django.shortcuts import render

from django.views.generic import ListView, DetailView, CreateView
from loginapp import forms
from django.urls import reverse_lazy

from loginapp.models import loginmodel
from django.contrib.auth.forms import UserCreationForm

# Create your views here.
class LoginView(CreateView):
    template_name = "new.html"
    form_class = forms.CustomUserCreationForm
    success_url = reverse_lazy("complete")


def signupcomplete(request):
    return HttpResponse("<p>sign upが完了しました </p>")

def loginfunction(request):
    return HttpResponse("<h1>hello world1</h1>")

class userlistview(ListView):
    model = loginmodel
    template_name = "userlist.html"

class userdetailview(DetailView):
    model = loginmodel
    template_name = "userdetail.html"

class signup(CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy("login")#登録成功時に遷移するurl
    template_name = "signup.html"
