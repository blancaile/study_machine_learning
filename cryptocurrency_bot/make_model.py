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


BUF = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\edit_btcusdt_f.csv")
EUF = pd.read_csv(r"C:\github_folder\study_machine_learning\cryptocurrency_bot\datasets\edit_ethusdt_f.csv")


BUF_feature = make_feature(BUF)
EUF_feature = make_feature(EUF)

#結合
feature = pd.concat([BUF_feature, EUF_feature], axis=1)



