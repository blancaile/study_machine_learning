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



#受け取ったデータから特徴量を生成
def make_feature(df):
    df["rsi9"] = ta.RSI(df["close"], timeperiod=9)
    #return df.dropna(how="any") #後でnanの個数を使う
    return df



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
    return eth_ratio.dropna(how="any")



#sliding windowを生成
#for i in range(window_size, len(feature)):  #重い遅い
#    data.append(feature[i - window_size: i, :])
#    #print(type(data))


#https://stackoverflow.com/questions/43185589/sliding-windows-from-2d-array-that-slides-along-axis-0-or-rows-to-give-a-3d-arra
def strided_axis0(a, L): 
    # INPUTS :
    # a is array
    # L is length of array along axis=0 to be cut for forming each subarray

    # Length of 3D output array along its axis=0
    nd0 = a.shape[0] - L + 1

    # Store shape and strides info
    m,n = a.shape
    s0,s1 = a.strides

    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(a, shape=(nd0,L,n), strides=(s0,s0,s1))




#ファイルの読み込み
BUF = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\edit_btcusdt_f.csv")
EUF = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\edit_ethusdt_f.csv")


#説明変数を生成
BUF_feature = make_feature(BUF)
EUF_feature = make_feature(EUF)


#目的変数を生成
mlater = 1 #何分後のup,downを予測するか
train= make_training_data(EUF["close"], mlater, 0.0001, -0.0001)
train = train.to_numpy()


#説明変数の結合
feature = pd.concat([BUF_feature, EUF_feature], axis=1)
nancount = len(feature[feature.isnull().any(axis=1)])

#print(nancount)
feature = feature.dropna(how="any").to_numpy()


#目的変数データ生成時にできたnanを含む行を説明変数データから削除
feature = np.delete(feature, [i for i in range(len(feature) - mlater, len(feature))], axis=0)


#標準化
feature = scipy.stats.zscore(feature)
#print(feature.shape)


#ウィンドウ作成
window_size = 1440


slide_feature = strided_axis0(feature, window_size)
#print(slide_feature)
print(slide_feature.shape)



#教師データと訓練データのサイズ調整
train = np.delete(train, [i for i in range(window_size + nancount -1)], axis=0)
#print("////////////////")
#print(train)
print(train.shape)



#モデル作成

#学習



