#https://tutorials.chainer.org/ja/09_Introduction_to_Scikit-learn.html

from sklearn.datasets import load_boston

dataset = load_boston()

x = dataset.data
t = dataset.target

print(x.shape)

print(t.shape)

from sklearn.model_selection import train_test_split

#テスト用データセットを全体の30%を用いて作成する, 残りの70%は訓練用に用いる
#サンプルの中からランダムに抽出して訓練する
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()

reg_model.fit(x_train, t_train)

print(reg_model.coef_)
print(reg_model.intercept_)

print(reg_model.score(x_train, t_train))

print(reg_model.predict(x_test[:1]))

print(t_test[0])

print(reg_model.score(x_test, t_test))

from sklearn.preprocessing import StandardScaler
#標準化(平均が0, 分散が1になるようにスケーリングする)を行う
scaler = StandardScaler()

scaler.fit(x_train)

#平均
print(scaler.mean_)

#分散
print(scaler.var_)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

reg_model = LinearRegression()

reg_model.fit(x_train_scaled, t_train)
print(reg_model.score(x_train_scaled, t_train))

print(reg_model.score(x_test_scaled, t_test))


from sklearn.preprocessing import PowerTransformer
#べき変換
scaler = PowerTransformer()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

reg_model = LinearRegression()
reg_model.fit(x_train_scaled, t_train)

print(reg_model.score(x_train_scaled, t_train))
print(reg_model.score(x_test_scaled, t_test))

from sklearn.pipeline import Pipeline

#パイプラインの作成
pipeline = Pipeline([
    ("scaler", PowerTransformer()),
    ("reg", LinearRegression())
])

pipeline.fit(x_train, t_train)

print(pipeline.score(x_train, t_train))

linear_result = pipeline.score(x_test, t_test)
print(linear_result)