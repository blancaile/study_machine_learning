#https://tutorials.chainer.org/ja/11_Introduction_to_Pandas.html

import pandas as pd

df = pd.read_csv("california_housing_train.csv")

print(type(df))

print(df)

print(df.head)

print(df.head(3))

print(df["longitude"].head(3))

#csvファイルを作成
df.to_csv("sample.csv")

print(df.shape)

print(df.mean())
print(df.var())

#各列のNone, NaN, Natのいずれでもない値の数
print(df.count())

print(df.describe)

#相関係数の算出
print(df.corr())

#列を昇順に並び替え
df_as = df.sort_values(by="total_rooms")

print(df_as.head())

#降順に並び替え
df_de = df.sort_values(by="total_rooms", ascending=False)

print(df_de.head())

print(df.head(3))

print(df.iloc[0, 0])
print(df.iloc[1, 1])

#すべての行の最後の列を選択
t = df.iloc[:, -1]
print(t.head(3))

#すべての行の先頭の列から末尾の列のひとつ手前まで選択
x = df.iloc[:, 0:-1]
print(x.head(3))

mask = df["median_house_value"] > 70000
#条件を満たすかどうかを表すTrue, Falseが格納される
print(mask.head())

print(df[mask].head())

mask2 = (df["median_house_value"] < 70000) | (df["median_house_value"] > 80000)
print(mask2.head())

print(df[mask2].head())

#新しいtargetの列をNoneで初期化
df["target"] = None
print(df.head())

mask1 = df['median_house_value'] < 60000
mask2 = (df['median_house_value'] >= 60000) & (df['median_house_value'] < 70000)
mask3 = (df['median_house_value'] >= 70000) & (df['median_house_value'] < 80000)
mask4 = df['median_house_value'] >= 80000

#列を名前で指定する場合はlocを用いる
df.loc[mask1, "target"] = 0
df.loc[mask2, "target"] = 1
df.loc[mask3, "target"] = 2
df.loc[mask4, "target"] = 3

print(df.head())

df.iloc[0, 0] = None
print(df.head(3))

#欠損値のある行を削除
df_dropna = df.dropna()
print(df_dropna.head(3))

mean = df.mean()
print(mean)

#欠損値をmeanで補完
df_fillna = df.fillna(mean)
print(df_fillna.head(3))

print(type(df))

print(type(df.values))

print(df.values)

print(type(df["longitude"]))

print(type(df["longitude"].values))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.DataFrame(
    data = np.random.randn(10, 10)
)

print(df)


figure, ax = plt.subplots()

#df.plot()

ax.plot(df)

figure.savefig("test.jpg")