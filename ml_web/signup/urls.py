from django.urls import path
from . import views

urlpatterns = [
    path("new/",views.signup.as_view(),name="signup"),
]