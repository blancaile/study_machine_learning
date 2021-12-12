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
from sklearn.metrics import accuracy_score
import datetime


#受け取ったデータから特徴量を生成
def make_feature(df):
    df = df.iloc[::-1] #反転
    df["datetime"] = pd.to_datetime(df["date"])
    df["minute"] = df["datetime"].dt.minute
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df.drop(["date","datetime","unix", "symbol", "Volume USDT"], axis=1, inplace=True)
    #print(df.head())
    
    df["rsi9"] = ta.RSI(df["close"], timeperiod=9)
    df["ma35"] = ta.MA(df["close"], timeperiod=35)
    df["wma5"] = ta.WMA(df["close"], timeperiod=5)
    df["wma20"] = ta.WMA(df["close"], timeperiod=20) 
    df["upperband"], df["middleband"], df["lowerband"] = ta.BBANDS(df["close"], timeperiod=20)
    df["sar"] = ta.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)
    df["adosc"] = ta.ADOSC(df["high"], df["low"], df["close"], df["Volume ETH"], fastperiod=3, slowperiod=10)
    df["trix"] = ta.TRIX(df["close"], timeperiod=10)
    #df["upperband"], df["middlebabd"]
    #return df.dropna(how="any") #後でnanの個数を使う
    #print(df)
    #return df.drop(["open", "high", "low", "close", "tradecount"], axis=1)
    #sys.exit()
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
def make_stride(x,window_size):
    #name = x.columns.tolist()
    y = pd.DataFrame(x)
    for i in range(window_size):
        y = pd.concat(
            [y, x.shift(i+1).add_suffix("_"+str(i+1))],
            axis=1
        )
    #print(y.head())
        print("\r"+str(i)+" / "+str(window_size), end="")
        
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


#ファイルの読み込み
#BUF = pd.read_csv(os.getcwd() + r"\cryptocurrency_bot\datasets\edit_btcusdt_f.csv")
EUF = pd.read_csv(os.getcwd() + r"\cryptocurrency_bot\datasets\ethusdt_f.csv")


#説明変数を生成
#BUF_feature = make_feature(BUF.iloc[int(len(BUF)/1.1):, :]) #データが入りきらないとき
#EUF_feature = make_feature(EUF.iloc[int(len(EUF)/3):, :])
EUF_feature = make_feature(EUF) #本当ならすべてのデータを取り込みたい

#目的変数を生成
mlater = 5 #何分後のup,downを予測するか


#train= make_training_data(EUF["close"].iloc[int(len(EUF)/1.1):], mlater, 0.0005, -0.0005)
train= make_training_data(EUF["close"].iloc[::-1].reset_index(drop=True), mlater, 0.0009, -0.0009) #順番とindexには気をつける
train.columns = ["train"]
#print(train)

#train = train.iloc[int(len(EUF)/3):]   #データが入りきらないとき
train2 = train.to_numpy()
u, counts = np.unique(train2, return_counts=True) #同じ出現率がよい
print(u)      #0,     1,    2
print(counts) #[164008 169315 165253]

#lightgbmはいらない
#one hot encoding
#train = to_categorical(train)

#sys.exit()


#説明変数の結合
#feature = pd.concat([BUF_feature, EUF_feature], axis=1)
feature = pd.DataFrame(EUF_feature)
nancount = len(feature[feature.isnull().any(axis=1)])

print(nancount)
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
window_size = 3


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
x = make_stride(concat_feature_train, window_size=window_size)

#print("feature shape is ", x.shape)
#sys.exit()




x_train, x_test, t_train, t_test = train_test_split(x.drop("train", axis=1), x["train"], test_size=0.1, random_state=0, shuffle=False)
x_train, x_eval, t_train, t_eval = train_test_split(x_train, t_train, test_size=0.1, random_state=0, shuffle=False)

lgb_train = lgb.Dataset(x_train, t_train)
lgb_eval = lgb.Dataset(x_eval, t_eval, reference=lgb_train)

params = {
    "task": "train",
    "boosting": "rf",
    "extra_trees": True,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_error",
    "device": "gpu",
    "bagging_freq": 1,
    "bagging_fraction": 0.9
}

#ValueError: Input numpy.ndarray must be 2 dimensional
model_lgb = lgb.train(params=params,train_set=lgb_train,verbose_eval=10,valid_sets=lgb_eval)

y_pred = model_lgb.predict(x_test,num_iteration=model_lgb.best_iteration)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
#print(t_test)

acc = sum(t_test == y_pred) / len(t_test)
print("acc: ",acc)



