# Generated by Django 4.0 on 2022-02-03 06:59

import CryptographicFields.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CustomUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('username', models.CharField(max_length=50, unique=True, verbose_name='username')),
                ('password', models.CharField(max_length=50, verbose_name='password')),
                ('api_key', CryptographicFields.fields.CharField(max_length=100, verbose_name='api_key')),
                ('secret_key', CryptographicFields.fields.CharField(max_length=100, verbose_name='secret_key')),
                ('is_active', models.BooleanField(default=True)),
                ('is_admin', models.BooleanField(default=False)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]