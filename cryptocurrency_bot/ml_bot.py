import datetime
from time import timezone
import ccxt
import os
from dateutil.tz import UTC
import lightgbm as lgb
from matplotlib.pyplot import axis
import make_model_lightgbm as mm
import pandas as pd
import numpy as np


def now_predict(COIN):
    day = 60*60*2 #1日前までのチャートを取得
    noweth = mm.getklinedata(COIN,day)
    date = pd.to_datetime(noweth["date"]).iloc[-1].replace(tzinfo=UTC).astimezone(datetime.timezone(datetime.timedelta(hours=9))).replace(tzinfo=None)
    print(str(date) + "から" + str(mm.mlater) + "分後のdown/stay/up確率を求めます")
    x = mm.make_feature(noweth.iloc[::-1], COIN)
    x["train"] = x["close"]
    x = mm.make_lag_feature(x, mm.window_size)
    x_now = x.tail(1).drop(["train"], axis=1)
    #print(x_now.isnull().values.sum()) #NaNカウント
    if __name__ == "__main__":
        best = lgb.Booster(model_file=os.getcwd() +r"\cryptocurrency_bot\model.txt")
    else:
        best = lgb.Booster(model_file=os.getcwd() +r"\..\cryptocurrency_bot\model.txt")
    ypred = best.predict(x_now,num_iteration=best.best_iteration)

    return ypred


def now_order(exchange, y, p, COIN):
    print(y)
    #手持ちUSDTを取得
    balance = exchange.fetch_balance()["USDT"]["free"]
    print("今の手持ちUSDTは" + str(balance))


    #手持ちUSDTとモデルの確率とケリー基準?から何USDT使うか決定
    price = balance / 10 #仮


    #現在ETH/USDT価格を取得
    #ticker = exchange.fetch_ticker(COIN+"/USDT")["last"]
    #print(ticker)

    bidorask = lambda a: "asks" if a == 2 else "bids"
    ob = exchange.fetch_order_book(COIN+"/USDT")[bidorask(y)][0][0] #bidsこの価格でなら買う(今より安い) #asksこの価格でなら売る(今より高い)
    #print(ob)

    #注文
    side = lambda a: "buy" if a == 2 else "sell"
    order = exchange.create_order(
        symbol = COIN+"/USDT",
        type = "limit",
        price = ob, #指値価格
        side = side(y),
        amount = price/ob
    )

    print(order)
    return order["id"]


def order(apikey, secretkey):
    exchange = ccxt.binanceusdm({
        "apiKey": apikey,
        "secret": secretkey,
    })


    #モデルで予測
    ypred = now_predict("ETH")
    #[[0.3230353  0.36189054 0.31507416]]
    print("ypred is ", ypred)
    y = np.argmax(ypred,axis=1)[0]
    print("0,1,2 is", y)#0,1,2
    print(ypred[0][y]) #一番高い予測の確率表示

    if y != 1:
        order_id = now_order(exchange, y, ypred[0][y], "ETH")



#{注文している間はループ
#現在ETH/USDT価格を取得
# 予測時の価格からx%以上up/downしたら損切り
# 
# x分後になったら注文を閉じる
# }




def main():
    #key読み込み
    apikey = ""
    secretkey = ""
    with open(os.getcwd() + r"\cryptocurrency_bot\binance_api_key.txt","r") as f:
        for line in f:
            data = line.strip().split("=")
            if data[0] == "api_key":
                apikey = data[1]
            elif data[0] == "secret_key":
                secretkey = data[1]
    f.close()

    order(apikey, secretkey)


#他のファイルから呼び出したとき実行しないようにするために書く
if __name__ == "__main__":
    main()