import ccxt
import os






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



exchange = ccxt.binanceusdm({
    "apiKey": apikey,
    "secret": secretkey,
})

#価格取得
ticker = exchange.fetch_ticker("ETH/USDT")

print(ticker["last"])

#口座取得
balance = exchange.fetch_balance()

print(balance["USDT"])




#モデルで予測

#手持ちUSDTを取得

#手持ちUSDTとモデルの確率とケリー基準?から何USDT使うか決定

#現在ETH/USDT価格を取得

#注文

#{注文している間はループ
#現在ETH/USDT価格を取得
# 予測時の価格からx%以上up/downしたら損切り
# 
# x分後になったら注文を閉じる
# }

