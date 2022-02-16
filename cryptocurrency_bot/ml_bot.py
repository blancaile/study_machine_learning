import datetime
from time import sleep, timezone
import ccxt
import os
from dateutil.tz import UTC
import lightgbm as lgb
from matplotlib.pyplot import axis
import make_model_lightgbm as mm
import pandas as pd
import numpy as np
import time
from retry import retry


@retry(tries=5,delay=1,backoff=2)
def create_order(exchange, symbol, type, side, amount, price, params = {}):#注文を出す
    order = exchange.create_order( #stop limit: side=buyのときは高く設定，sellは低く設定
        symbol = symbol,
        type = type,
        side = side,
        amount = amount,
        price = price,
        params = params
    )
    return order

@retry(tries=5,delay=1,backoff=2)
def fetch_order_book(exchange, COIN, bidorask):#COINの価格を取得
    ob = exchange.fetch_order_book(COIN+"/USDT")[bidorask][0][0] #bidsこの価格でなら買う(今より安い) #asksこの価格でなら売る(今より高い)
    return ob

@retry(tries=5,delay=1,backoff=2)
def fetch_balance(exchange):#手持ちのUSDTを取得
    balance = exchange.fetch_balance()["USDT"]["free"]
    return balance

@retry(tries=5,delay=1,backoff=2)#取引履歴が存在しない場合
def fetch_my_trades(exchange, COIN):#取引履歴を取得
    try:
        trades = exchange.fetch_my_trades(symbol = COIN + "/USDT")
    except ccxt.ExchangeError:
        pass
    return trades

@retry(tries=5,delay=1,backoff=2)#キャンセルできる注文が存在しない場合->無視して続行する
def cancel_order(exchange, id, COIN):#注文をキャンセルする
    try:
        c_order = exchange.cancel_order(id,symbol= COIN + "/USDT")#既にした注文をキャンセル
    except ccxt.ExchangeError:
        pass
    return c_order


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


    #p:yの確率
def now_order(exchange, y, p, COIN):
    print(y)
    #手持ちUSDTを取得
    balance = fetch_balance(exchange)
    print("今の手持ちUSDTは" + str(balance))


    #手持ちUSDTとモデルの確率とケリー基準?から何USDT使うか決定
    price = balance#仮

    #レバレッジ調整
    exchange.load_markets()
    market = exchange.markets["ETH/USDT"]
    exchange.fapiPrivate_post_leverage({
        "symbol": market["id"],
        "leverage": 1,
    })


    #現在ETH/USDT価格を取得
    #ticker = exchange.fetch_ticker(COIN+"/USDT")["last"] 
    #print(ticker)

    bidorask = lambda a: "asks" if a == 2 else "bids"

    #errorhandling未対応
    # ob = exchange.fetch_order_book(COIN+"/USDT")[bidorask(y)][0][0] #bidsこの価格でなら買う(今より安い) #asksこの価格でなら売る(今より高い)
    ob = fetch_order_book(exchange, COIN, bidorask(y))
    print(ob)

    #注文
    print("amount, ",price/ob)
    #if price/ob < 0.002: #エラー表示させる

    
    side = lambda a: "buy" if a == 2 else "sell"

    order = create_order(exchange, COIN+"/USDT", "limit", side(y), price/ob, ob, {})

    print(order)
    return order#["id"]


#予測と実行
def order(apikey, secretkey):
    exchange = ccxt.binanceusdm({
        "apiKey": apikey,
        "secret": secretkey,
        'options': {
        "defaultType": "future",
    }
    })


    while(True):#スイッチがオンの間
    #モデルで予測
        ypred = now_predict("ETH")
        #[[0.3230353  0.36189054 0.31507416]]
        print("ypred is ", ypred)
        y = np.argmax(ypred,axis=1)[0]
        print("0,1,2 is", y)#0,1,2
        print(ypred[0][y]) #一番高い予測の確率表示
        #y = 2#テスト用


        #予測結果がstayなら1分待つ
        if y == 1:
            print("sleep 60sec")
            sleep(60)

        #予測結果がup or downなら注文をする
        elif y != 1:
            order = now_order(exchange, y, ypred[0][y], "ETH")
            sleep(1)

            trades = fetch_my_trades(exchange, "ETH")
            print(" ")
            print(trades[-1])
            #trades = exchange.fetch_order_trades(order["id"],symbol="ETH/USDT")#fetch order trades はspot only

            if trades[-1]["order"] == order["id"]:#注文が通った
                print("ifになった")#一定時間ごとに価格を監視して損切り態勢、mm.mlater後にポジションを閉じる
                si = lambda a: "buy" if a == "sell" else "sell"#side逆転
                ch = lambda a: 1 if a == "buy" else -1#buyなら1,sellなら-1
                close_position = create_order(exchange, trades[-1]["symbol"], "limit", si(trades[-1]["side"]), trades[-1]["amount"], trades[-1]["price"] + 10 * ch(trades[-1]["side"]), {"reduceOnly": True})
                tre = fetch_my_trades(exchange, "ETH")
                while(close_position["id"] != tre[-1]["order"]):#close positionが通らない間
                    sleep(2)
                    nowtime = int(time.time())
                    flag = False
                    tre = fetch_my_trades(exchange, "ETH")
                    if mm.mlater * 60<= nowtime - trades[-1]["timestamp"] // 1000: #規定時間になったら
                        print(mm.mlater, "分経過")
                        while(True):#注文を閉じるまでトライ 
                            c_order = cancel_order(exchange, close_position["id"], "ETH")
                            print('キャンセル成功')
                            #noweth = exchange.fetch_ticker(symbol="ETH/USDT")
                            bidorask = lambda a: "bids" if a == 2 else "asks"
                            noweth = fetch_order_book(exchange, "ETH", bidorask(y))
                            print("now eth price is ", noweth)

                            fin_position = create_order(exchange, trades[-1]["symbol"], "limit", si(trades[-1]["side"]), trades[-1]["amount"], noweth, {"reduceOnly": True})
                            sleep(1)
                            ftrades = fetch_my_trades(exchange, "ETH")
                            if fin_position["id"] == ftrades[-1]["order"]:#close注文が通ったらbreak
                                flag = True
                                print("ポジションclose成功")
                                break
                    
                    if flag:
                        break


            elif trades[-1]["order"] != order["id"]: #注文が通らなかったら
                print("elifになった")
                c_order = cancel_order(exchange, order["id"], "ETH")
            else:
                print("elseになった")
                c_order = cancel_order(exchange, order["id"], "ETH")


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