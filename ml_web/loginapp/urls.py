from django.urls import path
from . import views

urlpatterns = [
    path("new/",views.signup.as_view(),name="signup"),
    path("signup/", views.signup.as_view(),name="signup"),
    path("complete/", views.signupcomplete,name="complete"),
    path("login/", views.loginfunction, name="hello"),
    path("list/", views.userlistview.as_view(), name="list"),
    path("detail/<int:pk>", views.userdetailview.as_view(), name="detail"),
]