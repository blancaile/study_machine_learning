from django.db import models
from django.contrib.auth.models import AbstractUser


# Create your models here.
class loginmodel(AbstractUser):
    #username = models.CharField(max_length = 100)
    password = models.CharField(max_length = 100)
    api_key = models.CharField(max_length = 100)
    secret_key = models.CharField(max_length = 100)