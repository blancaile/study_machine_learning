from operator import ilshift
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
import talib as ta
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
from tensorflow.python.keras.backend import print_tensor
from tensorflow.python.ops.array_ops import sequence_mask
import tensorflow_datasets as tfds
from sklearn.preprocessing import StandardScaler
import scipy.stats

def shaping():
    # open     high      low    close  Volume ETH   Volume USDT  tradecount
    ethusdtf = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\ethusdt_f.csv")
    ethusdtfsort = ethusdtf.sort_values("unix")
    ethusdtfsort = ethusdtfsort.reset_index(drop=True)
    ethusdtfsort.drop(["unix", "date", "symbol"], axis=1, inplace=True)
    ethusdtfsort.to_csv("edit_ethusdt_f.csv")




#受け取ったデータから特徴量を生成
def make_feature(df):
    df["rsi9"] = ta.RSI(df["close"], timeperiod=9)
    return df.dropna(how="any")



#up,stay,downの閾値とxを受け取りx分後のethのcloseの変化率を３値(1,0,-1)に分類する
def make_training_data(ed, x = 1, up = 0.0001, down = -0.0001):
    e_d = ed.to_dict() #dictionaryの方が早い

    for i in range(len(e_d) - x):
        
        ratio = (e_d[i+x] - e_d[i]) / e_d[i]
        #print("ratio is ", ratio)
        if ratio > up:
            e_d[i] = 1
        elif ratio < down:
            e_d[i] = -1
        else:
            e_d[i] = 0

    #print(type(e_d))
    #eth_ratio = pd.DataFrame(e_d, index=["i",]) #遅い
    eth_ratio = pd.DataFrame.from_dict(e_d, orient="index")
    eth_ratio.iloc[-x:] = np.nan #後ろからx分はNaN
    #print(eth_ratio)
    return eth_ratio

#1日分の
def slided_view2D(arr, N=1440):
    r, c = arr.shape
    rs, cs = arr.strides
    strided = np.lib.stride_tricks.as_strided
    return strided(arr, shape=(r-N+1, N, c), strides=(rs, rs, cs))


#読み込み
BUF = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\edit_btcusdt_f.csv")
EUF = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\edit_ethusdt_f.csv")

print("hello")
#説明変数を作成
BUF_feature = make_feature(BUF)
EUF_feature = make_feature(EUF)

window_size = 1440

#目的変数を作成
train= make_training_data(EUF["close"], 1, 0.0001, -0.0001)



#説明変数の結合
feature = pd.concat([BUF_feature, EUF_feature], axis=1)


#標準化
feature = feature.to_numpy()
feature = scipy.stats.zscore(feature)


#ウィンドウ作成
data = []
window_size = 1440



for i in range(window_size, len(feature)):  #重い遅い
    data.append(feature[i - window_size: i, :])
    #print(type(data))


slide_feature = np.array(data)
print(slide_feature)
print(slide_feature.shape)



#モデル作成

#学習



