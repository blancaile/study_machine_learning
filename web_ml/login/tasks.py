from celery import shared_task
import sys
sys.path.append("../cryptocurrency_bot/")
import ml_bot


@shared_task
def ml_order(apikey, secretkey, username):
    ml_bot.order(apikey=apikey, secretkey=secretkey, username = username)#非同期にする
