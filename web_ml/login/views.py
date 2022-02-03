from django.shortcuts import redirect, render
from . import forms
from . import models
# Create your views here.

def create(request):
    if(request.method == "GET"):
        model = models.CustomUser()
        form = forms.SignupForm()
        return render(request, "login/create.html",{"form":form, "model":model})
    
    elif(request.method == "POST"):
        print(request)
        username = request.POST["username"]
        #password = request.POST["password"]
        api_key = request.POST["api_key"]
        secret_key = request.POST["secret_key"]

        user = models.CustomUser(username = username,
        api_key = api_key, secret_key = secret_key)

        user.save()
        #user.set_password()
        return redirect(to="/admin")