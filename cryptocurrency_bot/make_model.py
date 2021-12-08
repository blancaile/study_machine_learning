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


#受け取ったデータから特徴量を生成
def make_feature(df):
    df["rsi9"] = ta.RSI(df["close"], timeperiod=9)
    df["ma35"] = ta.MA(df["close"], timeperiod=35)
    df["wma5"] = ta.WMA(df["close"], timeperiod=5)
    df["wma20"] = ta.WMA(df["close"], timeperiod=20) 
    df["upperband"], df["middleband"], df["lowerband"] = ta.BBANDS(df["close"], timeperiod=20)
    df["sar"] = ta.SAR(df["high"], df["low"], acceleration=0.02, maximum=0.2)
    df["adosc"] = ta.ADOSC(df["high"], df["low"], df["close"], df["Volume"], fastperiod=3, slowperiod=10)
    df["trix"] = ta.TRIX(df["close"], timeperiod=10)
    #df["upperband"], df["middlebabd"]
    #return df.dropna(how="any") #後でnanの個数を使う
    #print(df)
    return df.drop(["open", "high", "low", "close", "Volume", "tradecount"], axis=1)



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
EUF = pd.read_csv(os.getcwd() + r"\cryptocurrency_bot\datasets\edit_ethusdt_f.csv")


#説明変数を生成
#BUF_feature = make_feature(BUF.iloc[int(len(BUF)/1.1):, :]) #データが入りきらないとき
EUF_feature = make_feature(EUF.iloc[int(len(EUF)/3):, :])
#EUF_feature = make_feature(EUF)

#目的変数を生成
mlater = 5 #何分後のup,downを予測するか


#train= make_training_data(EUF["close"].iloc[int(len(EUF)/1.1):], mlater, 0.0005, -0.0005)
train= make_training_data(EUF["close"], mlater, 0.0013, -0.0013)
#print(train)

train = train.iloc[int(len(EUF)/3):]   #データが入りきらないとき
train = train.to_numpy()
u, counts = np.unique(train, return_counts=True) #同じ出現率がよい
print(u)      #0,     1,    2
print(counts) #[164008 169315 165253]
#one hot encoding
train = to_categorical(train)

#sys.exit()


#説明変数の結合
#feature = pd.concat([BUF_feature, EUF_feature], axis=1)
feature = pd.DataFrame(EUF_feature)
nancount = len(feature[feature.isnull().any(axis=1)])

#print(nancount)
feature = feature.dropna(how="any").to_numpy()


#目的変数データ生成時にできたnanを含む行を説明変数データから削除
feature = np.delete(feature, [i for i in range(len(feature) - mlater, len(feature))], axis=0)


#標準化
feature = scipy.stats.zscore(feature)
#print(feature.shape)


#ウィンドウ作成
window_size = 60*5


slide_feature = strided_axis0(feature, window_size)
print(slide_feature.shape)


#教師データと訓練データのサイズ調整
train = np.delete(train, [i for i in range(window_size + nancount - 1)], axis=0)
print(train.shape)



#x_train, x_test, t_train, t_test = train_test_split(slide_feature, train, test_size=0.1, random_state=0)


#モデル作成
input_dim = slide_feature.shape[2]
output_dim = 3
hidden_units = 10
learning_rate = 0.001
batch_size = 32
epoch = 50



model = Sequential()

model.add(tf.compat.v1.keras.layers.CuDNNLSTM(
    hidden_units,
    input_shape = (window_size, input_dim),
    return_sequences = False
))
#model.add(Dropout(0.3))
model.add(Dense(output_dim,  activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate), metrics=["accuracy"])
model.summary()


#学習
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(
    slide_feature, train,
    batch_size=batch_size,
    epochs=epoch,
    validation_split=0.1
)


# print("Prediction")
# predict = model.predict(x_test)
# j=0
# k=0
# for i, test in enumerate(predict):

#     if (t_test[i] == np.round(test)).all():
#         j+=1
#     k+=1

# print("正解率:",round(100 * j / x_test.shape[0], 1), "%")


modelplot(history)

