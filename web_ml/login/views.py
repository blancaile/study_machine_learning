from django.shortcuts import redirect, render
from . import forms
from . import models

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
        return redirect(to="/admin")