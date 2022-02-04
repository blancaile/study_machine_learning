#from django.contrib import admin
from django.urls import path, include
from . import views
urlpatterns = [
    #path('admin/', admin.site.urls),
    path("",views.temp, name="temp"),
    path("login/",views.signup, name="signup"),
    path("logout/",views.Logout,name="Logout"),
    path("create/",views.create, name="create"),
    path("home/",views.index, name="index"),
]
