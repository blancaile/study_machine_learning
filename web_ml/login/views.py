from audioop import reverse
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import redirect, render
from . import forms
from . import models
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout

from Crypto.Cipher import AES
import hashlib
import base64
from Crypto.Util.Padding import pad, unpad
# Create your views here.

def encrypt(data,password):
    K = hashlib.sha1(str(password).encode("utf-8")).hexdigest()
    Key = bytes.fromhex(K[:32])
    aes = AES.new(Key, AES.MODE_ECB)

    BLOCK_SIZE = 16
    data = data.encode("utf-8")
    data = base64.b64encode(data)
    data = aes.encrypt(pad(data,BLOCK_SIZE))
    return data
        
def decrypt(data,password):
    K = hashlib.sha1(str(password).encode("utf-8")).hexdigest()
    Key = bytes.fromhex(K[:32])
    aes = AES.new(Key, AES.MODE_ECB)

    data = aes.decrypt(data)
    data = base64.b64decode(data)
    data = data.decode("utf-8")
    return data

def create(request):
    if(request.method == "GET"):
        model = models.CustomUser()
        form = forms.SignupForm()
        return render(request, "login/create.html",{"form":form, "model":model})
    
    elif(request.method == "POST"):
        print(request)
        username = request.POST["username"]
        password = request.POST["password"] #なぜかパスワードが存在しない
        api_key = request.POST["api_key"]
        secret_key = request.POST["secret_key"]
       
        api_key = encrypt(api_key, password)
        secret_key = encrypt(secret_key, password)

        user = models.CustomUser(username = username,password=password,
        api_key = api_key, secret_key = secret_key)

        user.save()
        user.set_password(user.password)#ハッシュ化
        user.save()
        return redirect(to="signup")

def temp(request):
    return redirect(to="signup")


#def index(LoginRequiredMixin,request):
@login_required
def index(request):
    #print("request is ", request.user) #urlがリクエストされてる？
    #print("request is ", request.user.password)
    params = {"username":request.user.username,}
    return render(request, "login/home.html",context=params)

def signup(request):
    if request.method == "POST":
        id = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request,username=id, password=password)
        print(id, password, user)
        if user:
            if user.is_active:
                login(request,user)#ここでエラー
                return redirect(to="index")
            else:
                return HttpResponse("アカウントが有効ではありません")
        else:
            return HttpResponse("ログインIDかパスワードが間違っています")
    else:
        return render(request,"login/login.html")

    #     form = UserCreationForm(request.POST)
    #     if form.is_valid():
    #         user = form.save()
    #         login(request,user)
    #         return redirect("index/")
    # else:
    #     form = UserCreationForm()

    # return render(request, "login/login.html",{"form":form})

@login_required
def Logout(request):
    logout(request)
    return redirect(to="temp")