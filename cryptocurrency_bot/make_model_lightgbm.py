import os
import numpy as np
import pandas as pd
import talib as ta
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
import scipy.stats
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import sys
import lightgbm as lgb
#import optuna.integration.lightgbm as lgb #optuna
from sklearn.metrics import accuracy_score
import datetime
import time
import requests
import warnings
warnings.simplefilter("ignore")


#受け取ったデータから特徴量を生成
def make_feature(df, COIN):
    #考察
    #Overlap Studiesよりもモメンタムインジケーターの方が重要度が高い
    #ラグ特徴量はtrainが役に立っている,テクニカル指標のラグ特徴量はあまり役に立っていない   

    df = df.iloc[::-1] #反転
    df["datetime"] = pd.to_datetime(df["date"])
    #print(df.head())
    #df["minute"] = df["datetime"].dt.minute
    #df["hour"] = df["datetime"].dt.hour
    #df["dayofweek"] = df["datetime"].dt.dayofweek
    df.drop(["date","datetime","unix", "symbol", "Volume USDT"], axis=1, inplace=True)
    #print(df.head())

    f = [5,8,13,21,34,55,89,144]#,233
    for i in f:
        df["rsi"+str(i)] = ta.RSI(df["close"], timeperiod=i)#重要
        #df["ma"+str(i)] = ta.MA(df["close"], timeperiod=i)
        #df["dema"+str(i)] = ta.DEMA(df["close"], timeperiod=i)
        #df["sma"+str(i)] = ta.SMA(df["close"], timeperiod=i)
        #df["ema"+str(i)] = ta.EMA(df["close"], timeperiod=i)
        #df["wma"+str(i)] = ta.WMA(df["close"], timeperiod=i)
        #df["kama"+str(i)] = ta.KAMA(df["close"], timeperiod=i)
        #df["tema"+str(i)] = ta.TEMA(df["close"], timeperiod=i)
        #df["trima"+str(i)] = ta.TRIMA(df["close"], timeperiod=i)
        #df["upperband"+str(i)], df["middleband"+str(i)], df["lowerband"+str(i)] = ta.BBANDS(df["close"], timeperiod=i)
        #df["trix"+str(i)] = ta.TRIX(df["close"], timeperiod=i)
        #df["tsf"+str(i)] = ta.TSF(df["close"], timeperiod=i)
        #df["adxr"+str(i)] = ta.ADXR(df["high"], df["low"], df["close"], timeperiod=i)
        df["natr"+str(i)] = ta.NATR(df["high"], df["low"], df["close"], timeperiod=i)#重要
        df["cci"+str(i)] = ta.CCI(df["high"],df["low"],df["close"],timeperiod=i)#重要
        #df["aroonosc"+str(i)] = ta.AROONOSC(df["high"],df["low"],timeperiod=i)
        df["cmo"+str(i)] = ta.CMO(df["close"],timeperiod=i)#重要
        #df["dx"+str(i)] = ta.DX(df["high"],df["low"],df["close"],timeperiod=i)
        df["mfi"+str(i)] = ta.MFI(df["high"],df["low"],df["close"],df["Volume "+COIN],timeperiod=i)
        df["mom"+str(i)] = ta.MOM(df["close"],timeperiod=i)#すこし重要?

    #df["sar"] = ta.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)
    df["adosc"] = ta.ADOSC(df["high"], df["low"], df["close"], df["Volume "+COIN], fastperiod=3, slowperiod=10)
    df["obv"] = ta.OBV(df["close"],df["Volume "+COIN])#すこし重要?
    #df["ht_trendline"] = ta.HT_TRENDLINE(df["close"])
    df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["bop"] = ta.BOP(df["open"],df["high"],df["low"],df["close"]) #重要度１位
    
    














    # df["rsi9"] = ta.RSI(df["close"], timeperiod=9)

    # df["ma10"] = ta.MA(df["close"], timeperiod=10)
    # df["ma35"] = ta.MA(df["close"], timeperiod=35)

    # df["ema5"] = ta.EMA(df["close"], timeperiod=5)
    # df["ema8"] = ta.EMA(df["close"], timeperiod=8)
    # df["ema13"] = ta.EMA(df["close"], timeperiod=13)
    # df["ema200"] = ta.EMA(df["close"], timeperiod=200)

    # df["wma10"] = ta.WMA(df["close"], timeperiod=10)
    # df["wma20"] = ta.WMA(df["close"], timeperiod=20)
    # df["wma150"] = ta.WMA(df["close"], timeperiod=150)

    # df["upperband"], df["middleband"], df["lowerband"] = ta.BBANDS(df["close"], timeperiod=20)
    # df["sar"] = ta.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)
    # df["adosc"] = ta.ADOSC(df["high"], df["low"], df["close"], df["Volume "+COIN], fastperiod=3, slowperiod=10)
    # df["trix"] = ta.TRIX(df["close"], timeperiod=20)

    #df["upperband"], df["middlebabd"]
    #return df.dropna(how="any") #後でnanの個数を使う
    #print(df)
    #return df.drop(["open", "high", "low", "close", "tradecount"], axis=1)
    #sys.exit()
    df.drop(["close","open","high","low"], axis=1, inplace=True)
    return df.add_suffix("_"+COIN).reset_index(drop=True)



#up,stay,downの閾値とxを受け取りx分後のethのcloseの変化率を３値(2,1,0)に分類する
def make_training_data(ed, x = 1, up = 0.0001, down = -0.0001):
    e_d = ed.to_dict() #dictionaryの方が早い
    #print(e_d)
    #print(e_d[100], " ", e_d[101], " ", (e_d[101] - e_d[100])/e_d[100])
    for i in range(len(e_d) - x):
        
        ratio = (e_d[i+x] - e_d[i]) / e_d[i]
        #print("ratio is ", ratio)
        if ratio > up:
            e_d[i] = 2
        elif ratio < down:
            e_d[i] = 0
        else:
            e_d[i] = 1


    #print(type(e_d))
    #eth_ratio = pd.DataFrame(e_d, index=["i",]) #遅い
    eth_ratio = pd.DataFrame.from_dict(e_d, orient="index")
    eth_ratio.iloc[-x:] = np.nan #後ろからx分はNaN
    #print(eth_ratio)
    #return eth_ratio.dropna(how="any")
    return eth_ratio.reset_index(drop=True)



#window_sizeまでのラグ特徴量を生成
def make_lag_feature(x,window_size):
    #name = x.columns.tolist()
    print("window_size: ",window_size)
    y = pd.DataFrame(x)
    for i in range(window_size):
        y = pd.concat(
            [y, x.shift(i+1).add_suffix("_"+str(i+1))],
            axis=1
        )
    #print(y.head())
        print("\r"+str(i+1)+" / "+str(window_size), end="")
        
    print("")
    return y



#sliding windowを生成
#for i in range(window_size, len(feature)):  #重い遅い
#    data.append(feature[i - window_size: i, :])
#    #print(type(data))



#結果の描画
def modelplot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



#binance apiからcoin futuresの１日前からのklinesを持ってくる
def getklinedata(coin,day):
    sstamps=int(time.time()-day)*1000
    #estamps=int(time.time())*1000

    url="https://fapi.binance.com/fapi/v1/klines?symbol="+ coin + "USDT&interval=1m&startTime="+str(sstamps)#+"&endTime="+str(estamps)
    res = requests.get(url)
    res=res.json()
    #print(res)
    df = pd.DataFrame(res).drop([6,7,9,10,11],axis=1)
    df = df.set_axis(["date","open","high","low","close","Volume "+coin,"tradecount"],axis=1).astype(float)
    df["date"] = pd.to_datetime(df["date"],unit="ms")


    df["unix"] = df["date"]
    df["symbol"] = df["date"]
    df["Volume USDT"] = df["date"]

    return df


#ファイルの読み込み
BUF = pd.read_csv(os.getcwd() + r"\cryptocurrency_bot\datasets\btcusdt_f.csv")
EUF = pd.read_csv(os.getcwd() + r"\cryptocurrency_bot\datasets\ethusdt_f.csv")


#説明変数を生成
#BUF_feature = make_feature(BUF.iloc[:int(len(BUF)/1), :],"BTC") #データが入りきらないとき
#EUF_feature = make_feature(EUF.iloc[:int(len(EUF)/1), :],"ETH")
BUF_feature = make_feature(BUF,"BTC")
EUF_feature = make_feature(EUF,"ETH")
#print("BUF shape is", BUF_feature.shape)
#print("EUF shape is", EUF_feature.shape)
#feature = make_feature(EUF,"ETH") #本当ならすべてのデータを取り込みたい

#目的変数を生成
mlater = 10 #何分後のup,downを予測するか


#train= make_training_data(EUF["close"].iloc[int(len(EUF)/1.1):], mlater, 0.0005, -0.0005)
threshold = 0.00125 #閾値 #10
train= make_training_data(EUF["close"].iloc[::-1].reset_index(drop=True), mlater, threshold, -1 * threshold) #順番とindexには気をつける
train.columns = ["train"]
#print(train)
del(EUF)
del(BUF)
#train = train.iloc[int(len(EUF)/3):]   #データが入りきらないとき
train2 = train.to_numpy()
u, counts = np.unique(train2, return_counts=True) #同じ出現率がよい
print(u)      #0,     1,    2
print(counts) #[164008 169315 165253]
del(train2)
#lightgbmはいらない
#one hot encoding
#train = to_categorical(train)

#sys.exit()


#説明変数の結合
feature = pd.concat([EUF_feature, BUF_feature], axis=1)
#feature = pd.DataFrame(EUF_feature)
nancount = len(feature[feature.isnull().any(axis=1)])

#print(nancount)
#feature = feature.dropna(how="any").to_numpy()
 
#feature = feature.dropna(how="any") #///


#目的変数データ生成時にできたnanを含む行を説明変数データから削除
#slide_feature = np.delete(feature, [i for i in range(len(feature) - mlater, len(feature))], axis=0) #ndarray
#slide_feature = feature.drop(i for i in range(len(feature) - mlater, len(feature))) #dataframe #///

#print(nancount)
#標準化
#feature = scipy.stats.zscore(feature)
#slide_feature = scipy.stats.zscore(feature)
#print("feature shape is ",feature.shape)


#ウィンドウ作成
window_size = 5 #ラグ特徴量はあったほうがいい0.5->0.75
#10分後予測 size60 acc0.732
#          size1   acc0.742
#           size3   acc0.746
#           size5  acc0.750
#           size10  acc0.73
#           size35  acc0.73

#slide_feature = strided_axis0(feature, window_size)
#print("feature shape is ",feature.shape)


#教師データと訓練データのサイズ調整
#train = np.delete(train, [i for i in range(window_size + nancount - 1)], axis=0)
#train = np.delete(train, [i for i in range(nancount)], axis=0) #ndarray, lightgbm
#train = train.drop(i for i in range(nancount))
#print("train shape is ",train.shape)



concat_feature_train = pd.concat([train, feature], axis=1)
#print(concat_feature_train)
concat_feature_train.dropna(how="any", inplace=True)
concat_feature_train.reset_index(drop=True, inplace=True)



#ラグ特徴量を生成
x = make_lag_feature(concat_feature_train, window_size) #NaNどうしてる？
#print(x)
x = x.dropna(how="any",inplace=False).reset_index(drop=True)
#print(x)
print("feature shape is ", x.shape)
#sys.exit()




x_train, x_test, t_train, t_test = train_test_split(x.drop("train", axis=1), x["train"], test_size=0.1, random_state=0, shuffle=False)
del(x)
x_train, x_eval, t_train, t_eval = train_test_split(x_train, t_train, test_size=0.1, random_state=0, shuffle=False)
t_train_np = t_train.to_numpy()
u, counts = np.unique(t_train_np, return_counts=True) #同じ出現率がよい
print(u)      #0,     1,    2
print(counts) #[164008 169315 165253]


lgb_train = lgb.Dataset(x_train, t_train)
lgb_eval = lgb.Dataset(x_eval, t_eval, reference=lgb_train)
#print(x_train.columns)
#print(t_train.columns)

params = {
    "task": "train",
    "boosting": "rf",
    "extra_trees": True,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "device": "gpu",
    "bagging_freq": 1,
    "bagging_fraction": 0.9,
    'feature_pre_filter': False,
    "lambda_l1": 3.946722917499177e-05,
    "lambda_l2": 4.6903285635226136e-07,
    "num_levels": 244,
    "feature_fraction": 1.0,
    "min_child_samples": 5,
    "num_iterations": 50,
}

best_params, tuning_history = dict(), list()
model_lgb = lgb.train(params=params,train_set=lgb_train,verbose_eval=10,valid_sets=lgb_eval,num_boost_round=1000)
#Best Params: {'task': 'train', 'boosting': 'rf', 'extra_trees': True, 'objective': 'multiclass', 'num_class': 3, 
# 'metric': 'multi_logloss', 'device': 'gpu', 'bagging_freq': 1, 'bagging_fraction': 0.9, 'feature_pre_filter': False, 
# 'lambda_l1': 3.946722917499177e-05, 'lambda_l2': 4.6903285635226136e-07, 'num_leaves': 244, 'feature_fraction': 1.0,
#  'min_child_samples': 5, 'num_iterations': 50, 'early_stopping_round': None}

#print("Best Params:", model_lgb.params)
#print("Tuning history:", tuning_history)

model_lgb.save_model(os.getcwd() +r"\cryptocurrency_bot\model.txt") #モデル保存

y_pred = model_lgb.predict(x_test,num_iteration=model_lgb.best_iteration)
y_pred = np.argmax(y_pred, axis=1)


acc = sum(t_test == y_pred) / len(t_test)
print(len(t_test))
print("acc: ",acc)


lgb.plot_importance(model_lgb, figsize=(12, 6))
plt.show()

#保存したモデルの呼び出し
# best = lgb.Booster(model_file=os.getcwd() +r"\cryptocurrency_bot\model.txt")
# ypred = best.predict(x_test,num_iteration=best.best_iteration)
# ypred = np.argmax(ypred,axis=1)

# acc = sum(t_test == ypred) / len(t_test)
# print(len(t_test))
# print("acc: ",acc)


#ccxtを使用
#import ccxt

#key読み込み
# apikey = ""
# secretkey = ""
# with open(os.getcwd() + r"\cryptocurrency_bot\binance_api_key.txt","r") as f:
#     for line in f:
#         data = line.strip().split("=")
#         if data[0] == "api_key":
#             apikey = data[1]
#         elif data[0] == "secret_key":
#             secretkey = data[1]
# f.close()





# exchange = ccxt.binanceusdm({
#     "apikey": apikey,
#     "secret": secretkey,
# })

# info = exchange.fetch_ticker(symbol="ETH/USDT")
#print(info)



#今から一日前のデータでテスト

day = 60*60*24*2 #1日前までのチャートを取得
noweth = getklinedata("ETH",day)

time.sleep(1)
nowbtc = getklinedata("BTC",day)
print("noweth shape is ",noweth.shape)

NBUF_feature = make_feature(nowbtc.iloc[::-1],"BTC")
NEUF_feature = make_feature(noweth.iloc[::-1],"ETH")
print("nEUFF shape is ",NEUF_feature.shape)

nowtrain= make_training_data(noweth["close"].reset_index(drop=True), mlater, threshold, -1 * threshold)
nowtrain.columns = ["train"]
print("nowtrain shape is ",nowtrain.shape)

nowfeature = pd.concat([NEUF_feature, NBUF_feature], axis=1)
nowconcat_feature_train = pd.concat([nowtrain, nowfeature], axis=1)
print("nowconcat shape is ",nowconcat_feature_train.shape)

nowconcat_feature_train.dropna(how="any", inplace=True)
nowconcat_feature_train.reset_index(drop=True, inplace=True)
x2 = make_lag_feature(nowconcat_feature_train, window_size)
print("x2 shape is ",x2.shape)
x2 = x2.dropna(how="any").reset_index(drop=True) #これ怪しい
x_now = x2.drop("train", axis=1)
#print("xnow is \n",x_now)
t_now = x2["train"]
#print("t_now is \n",t_now)
best = lgb.Booster(model_file=os.getcwd() +r"\cryptocurrency_bot\model.txt")
ypred = best.predict(x_now,num_iteration=best.best_iteration)
ypred = np.argmax(ypred,axis=1)

acc = sum(t_now == ypred) / len(t_now)
print(len(t_now))
print("acc: ",acc)