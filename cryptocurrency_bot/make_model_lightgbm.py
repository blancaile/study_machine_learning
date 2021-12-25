import os
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import lightgbm as lgb
#import optuna.integration.lightgbm as lgb #optuna
from sklearn.metrics import accuracy_score
import time
import requests
import warnings
warnings.simplefilter("ignore")
from hurst import compute_Hc


#移動平均乖離率
def estrangement_rate(close, es):

    df = pd.DataFrame()
    df["es"] = (close - es)*100/es
    return df


#受け取ったデータから特徴量を生成
def make_feature(df, COIN):
    #移動平均よりもモメンタムインジケーターの方が重要度が高い,実数値よりも割合で表したほうがいい
    #ラグ特徴量はtrain(目的変数)が役に立っている,テクニカル指標のラグ特徴量はあまり役に立っていない
    #trainには未来の情報が含まれている,trainは使用してはならない

    df = df.iloc[::-1] #反転
    df["datetime"] = pd.to_datetime(df["date"])
    #df["minute"] = df["datetime"].dt.minute
    #df["hour"] = df["datetime"].dt.hour
    #df["dayofweek"] = df["datetime"].dt.dayofweek
    df.drop(["date","datetime","unix", "symbol", "Volume USDT"], axis=1, inplace=True)


    
    df["high-low"] = df["high"] - df["low"]
    #df["sar"] = ta.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)
    df["adosc"] = ta.ADOSC(df["high"], df["low"], df["close"], df["Volume "+COIN], fastperiod=3, slowperiod=10)
    df["obv"] = ta.OBV(df["close"],df["Volume "+COIN])#すこし重要?->重要度１位になった
    #df["ht_trendline"] = ta.HT_TRENDLINE(df["close"])
    df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["bop"] = ta.BOP(df["open"],df["high"],df["low"],df["close"]) #未来の情報を含むと重要度が１位になる
    df["trange"] = ta.TRANGE(df["high"],df["low"],df["close"])#
    df["TP"] = df[["high","low","close"]].mean(axis=1)
    df["VOLATILITY"] = df["trange"] / (df["TP"] * 100)
    #df["HurstExponent"]= compute_Hc(df["close"])[0]
    #df["HurstExponent2"] = df['close'].rolling(101).apply(lambda x: compute_Hc(x)[0])#ハースト指数


    #print(df.head(150))
    #sys.exit()
    #f=[7,14,30]
    f=[2,4,8,16,32]
    for i in f:
        df["trix"+str(i)] = ta.TRIX(df["close"], timeperiod=i)
        df["roc"+str(i)] = ta.ROC(df["close"],timeperiod=i)
        df["MAER"+str(i)] = estrangement_rate(df["close"], ta.SMA(df["close"],timeperiod=i)) #移動平均線乖離率
        df["mom"+str(i)] = ta.MOM(df["close"],timeperiod=i)
        df["rsi"+str(i)] = ta.RSI(df["close"], timeperiod=i)
        df["atr"+str(i)] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=i)
        df["ATRER"+str(i)] = estrangement_rate(df["close"], df["atr"+str(i)]) #ATR移動平均線乖離率
        df["cmo"+str(i)] = ta.CMO(df["close"],timeperiod=i)
        df["aroondown"+str(i)], df["aroonup"+str(i)] = ta.AROON(df["high"],df["low"],timeperiod=i)
        df["mfi"+str(i)] = ta.MFI(df["high"],df["low"],df["close"],df["Volume "+COIN],timeperiod=i)
        df["willr"+str(i)] = ta.WILLR(df["high"],df["low"],df["close"],timeperiod=i)
        df["atr"+str(i)] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=i)
        df["natr"+str(i)] = ta.NATR(df["high"], df["low"], df["close"], timeperiod=i)
        df["stddev"+str(i)] = ta.STDDEV(df["close"], timeperiod=i, nbdev=1)#標準偏差
        df["HL/STD"+str(i)] = df["high-low"] / df["stddev"+str(i)]#(高値-安値)/標準偏差
        df["pct_change"+str(i)] = df["close"].pct_change(periods=i)#騰落率
        df["stddevPCT"+str(i)] = ta.STDDEV(df["pct_change"+str(i)], timeperiod=i, nbdev=1)#騰落率の標準偏差

    #for clm in df:
    #    for i in f:
    #        df["ROCP_"+str(i)+clm] = ta.ROC(df[clm],timeperiod=i)

    df.drop(["open","high","low"], axis=1, inplace=True)
    #return df.add_suffix("_"+COIN).reset_index(drop=True)
    return df.reset_index(drop=True)


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
            #"minute","hour","dayofweek"
            [y, x.drop(["train","bop","macd","macdsignal","macdhist","adosc"], axis=1).shift(i+1).add_suffix("_"+str(i+1))],
            #[y, x.shift(i+1).add_suffix("_"+str(i+1))], #未来の情報を落とさないといけない
            axis=1
        )
    #print(y.head())
        print("\r"+str(i+1)+" / "+str(window_size), end="")
        
    print("")
    return y


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

    #print(df)
    df["unix"] = df["date"]
    df["symbol"] = df["date"]
    df["Volume USDT"] = df["date"]

    return df


#チャートデータ,コインの名前,何分後を予測するか,閾値,ラグ特徴量の生成数を受け取って目的変数と説明変数の結合データを返す
def make_data(eth, COIN, mlater, threshold, window_size):


    #説明変数を生成
    EUF_feature = make_feature(eth,COIN)
    #print("nEUFF shape is ",EUF_feature.shape)


    #目的変数を生成
    train= make_training_data(eth["close"].reset_index(drop=True), mlater, threshold, -1 * threshold) #順番とindexには気をつける
    train.columns = ["train"]
    #print("nowtrain shape is ",train.shape)


    #各目的変数のカウント
    train2 = train.to_numpy()
    u, counts = np.unique(train2, return_counts=True) #同じ出現率がよい
    print(u)      #0,     1,    2
    print(counts) #[164008 169315 165253]
    

    #説明変数と目的変数の結合
    concat_feature_train = pd.concat([train, EUF_feature], axis=1)
    #print("concat_feature_train shape is ",concat_feature_train.shape)


    #NaNの除去
    concat_feature_train.dropna(how="any", inplace=True)
    concat_feature_train.reset_index(drop=True, inplace=True)
    #print("DROP nowconcat_feature_train shape is ",concat_feature_train.shape)


    #ラグ特徴量の生成
    x = make_lag_feature(concat_feature_train, window_size)
    #x = pd.DataFrame(nowconcat_feature_train) #ラグ特徴量要らないときはこれ
    #print("x2 shape is ",x.shape)


    return x.dropna(how="any").reset_index(drop=True)


def main():
    #ファイルの読み込み
    EUF = pd.read_csv(os.getcwd() + r"\cryptocurrency_bot\datasets\ethusdt_f.csv") #timestamp降順
    #BUF = pd.read_csv(os.getcwd() + r"\cryptocurrency_bot\datasets\btcusdt_f.csv")


    mlater = 5 #何分後のup,downを予測するか
    threshold = 0.001 #閾値 #10 0.00125 #5 0.001
    window_size = 1 #ラグ特徴量はあったほうがいいacc0.5->0.75


    #目的変数と説明変数の生成
    x = make_data(EUF,"ETH",mlater,threshold,window_size)


    #データの分割
    x_train, x_test, t_train, t_test = train_test_split(x.drop("train", axis=1), x["train"], test_size=0.05, random_state=0, shuffle=False)
    x_train, x_eval, t_train, t_eval = train_test_split(x_train, t_train, test_size=0.1, random_state=0, shuffle=False)


    #分割後の各目的変数のカウント
    t_train_np = t_train.to_numpy()
    u, counts = np.unique(t_train_np, return_counts=True)
    print(u)      #0,     1,    2
    print(counts) #[164008 169315 165253]


    lgb_train = lgb.Dataset(x_train, t_train)
    lgb_eval = lgb.Dataset(x_eval, t_eval, reference=lgb_train)


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
        "learning_rate": 0.01
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


    #今から一日前までのデータでテスト

    day = 60*60*24*1 #1日前までのチャートを取得
    noweth = getklinedata("ETH",day)

    #訓練したモデルの読み込み
    best = lgb.Booster(model_file=os.getcwd() +r"\cryptocurrency_bot\model.txt")


    x = make_data(noweth.iloc[::-1],"ETH",mlater,threshold,window_size)
    #目的変数と説明変数の生成
    x_now = x.drop("train",axis=1)
    t_now = x["train"]


    ypred = best.predict(x_now,num_iteration=best.best_iteration)
    ypred = np.argmax(ypred,axis=1)


    acc = sum(t_now == ypred) / len(t_now)
    print(len(t_now))
    print("acc: ",acc)


#他のファイルから呼び出したとき実行しないようにするために書く
if __name__ == "__main__":
    main()
