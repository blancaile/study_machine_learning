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

#import cryptocurrency_bot.ml_bot as ml
import sys
sys.path.append("../cryptocurrency_bot/")
import ml_bot
#import cryptocurrency_bot
# Create your views here.

def encrypt(data,password):
    K = hashlib.sha1(str(password).encode("utf-8")).hexdigest()
    Key = bytes.fromhex(K[:32])
    aes = AES.new(Key, AES.MODE_ECB)

    BLOCK_SIZE = 16
    data = data.encode("utf-8")
    data = base64.b64encode(data)
    data = aes.encrypt(pad(data,BLOCK_SIZE)) #<class 'bytes'>
    #print(type(data))
    return data
        
def decrypt(data,password):
    K = hashlib.sha1(str(password).encode("utf-8")).hexdigest()
    Key = bytes.fromhex(K[:32])
    aes = AES.new(Key, AES.MODE_ECB)
    data = aes.decrypt(data)#error
    data = base64.b64decode(data)
    data = data.decode("utf-8")
    return data

def getexe(username):
    user = models.CustomUser.objects.get(username=username)
    return user.execute

#アカウント登録
def create(request):
    if(request.method == "GET"):
        model = models.CustomUser()
        form = forms.SignupForm()
        return render(request, "login/create.html",{"form":form, "model":model})
    
    elif(request.method == "POST"):
        #print(request)
        username = request.POST["username"]
        password = request.POST["password"] #なぜかパスワードが存在しない
        api_key = request.POST["api_key"]
        secret_key = request.POST["secret_key"]
       
        api_key = encrypt(api_key, password)#apiKey type is  <class 'bytes'>
        #print("apiKey type is ", type(api_key))
        #print(api_key)
        #api_key = api_key.decode("utf-8")#error
        #print("2apiKey type is ", type(api_key))
        secret_key = encrypt(secret_key, password)

        user = models.CustomUser(username = username,password=password,
        api_key = api_key, secret_key = secret_key)

        user.save()
        user.set_password(user.password)#ハッシュ化
        user.save()
        return redirect(to="signup")

def temp(request):
    return redirect(to="signup")

@login_required
def exefalse(request):
    user = models.CustomUser.objects.get(username=request.POST.get("username"))
    user.execute = False
    user.save()
    params = {"username":user.username,
            "execute":user.execute,
            "password":None,
            }
    #return render(request, "login/home.html",context=params)
    return redirect(to="index")

@login_required
def exetrue(request):
    user = models.CustomUser.objects.get(username=request.POST.get("username"))
    user.execute = True
    user.save()
    params = {"username":user.username,
            "execute":user.execute,
            "password":user.password,
            }
    #return render(request, "login/home.html",context=params)
    return redirect(to="index")#リダイレクト時にGETになる

#ホーム画面
@login_required
def index(request):
    user = models.CustomUser.objects.get(username=request.user.username)

    if request.method == "POST" and user.execute == False:#実行ボタンが表示される
        #ここでパスワードチェックを入れる
        password_check = authenticate(request, username=user.username, password=request.POST.get("password"))

        if password_check:
            apikey=decrypt(user.api_key, request.POST.get("password"))
            secretkey=decrypt(user.secret_key, request.POST.get("password"))
            
            user.execute = True
            user.save()
            params = {"username":user.username,
            "execute":user.execute,
            }
            
            ml_bot.order(apikey=apikey, secretkey=secretkey, username = user.username)#非同期にする
            return render(request, "login/home.html",context=params)
        else:
            return HttpResponse("パスワードが間違っています")
    elif request.method == "POST" and user.execute:#停止ボタンが表示される

        user.execute = False
        user.save()
        params = {"username":request.user.username,
        "execute":request.user.execute,
        }

        return render(request, "login/home.html",context=params)
    
    elif request.method == "GET":
        params = {"username":request.user.username,
        "execute":request.user.execute,
        }
        return render(request, "login/home.html",context=params)


#ログイン
def signup(request):
    if request.method == "POST":
        id = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request,username=id, password=password)
        #print(id, password, user)
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

#ログアウト
@login_required
def Logout(request):
    logout(request)
    return redirect(to="temp")