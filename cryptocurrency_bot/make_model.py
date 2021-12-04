import numpy as np
import pandas as pd
import os
import talib as ta



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
def make_training_data(df, x = 1, up = 0.0001, down = -0.0001):
    e_d = df.to_dict()

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


BUF = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\edit_btcusdt_f.csv")
EUF = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\edit_ethusdt_f.csv")


BUF_feature = make_feature(BUF)
EUF_feature = make_feature(EUF)
train = make_training_data(EUF["close"], 1, 0.0001, -0.0001)

print(train)
#結合
feature = pd.concat([BUF_feature, EUF_feature], axis=1)



