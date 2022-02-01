from django.db import models
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser, UserManager
from CryptographicFields import fields
# Create your models here.


class UserManager(BaseUserManager):
    def create_user(self,username,api_key,secret_key,password=None):
        if not username:
            raise ValueError("ユーザー名は必須です")
        if not api_key:
            raise ValueError("API KEYは必須です")
        if not secret_key:
            raise ValueError("SECRET API KEYは必須です")

        user = self.model(
            username = username,
            api_key = api_key,
            secret_key = secret_key,
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self,username,api_key,secret_key,password=None):
        user = self.create_user(
            username=username,
            password=password,
            api_key = api_key,
            secret_key = secret_key,
        )
        user.is_admin = True
        user.save(using=self._db)
        return user


class CustomUser(AbstractBaseUser):
    username = models.CharField(verbose_name="username",max_length=50,unique=True)
    password = models.CharField(verbose_name="password",max_length=50)
    api_key = fields.CharField(verbose_name="api_key",max_length=100)
    secret_key = fields.CharField(verbose_name="secret_key",max_length=100)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = "username"
    REQUIRED_FIEDS = ["username","password"]

    def __str__(self):
        return self.username

    def has_perm(self,perm,obj=None):
        return True

    def has_module_perms(self,app_label):
        return True

    @property
    def is_staff(self):
        return self.is_admin