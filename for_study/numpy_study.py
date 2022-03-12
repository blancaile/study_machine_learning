#https://tutorials.chainer.org/ja/08_Introduction_to_NumPy.html

from numpy.core.fromnumeric import shape
import tensorflow as tp
import numpy as np


a=np.array([1, 2, 3])

print(a)
print("shape=", a.shape, "\nrank=", a.ndim)

b = np.array(
    [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]
)

print(b)
print("shape=", b.shape, "\nrank=", b.ndim)
print("size=", b.size)

#要素がすべて0のndarrayを作成
c = np.zeros((3, 3))
print(c)

#要素がすべて1のndarrayを作成
d = np.ones((2, 3))
print(d)

#要素を指定した値で埋めたndarrayを作成
e = np.full((3, 2), 9)
print(e)

#指定したサイズの単位行列のndarrayを作成
f = np.eye(5)
print(f)

#要素を0-1間の乱数で埋めたndarrayを作成
g = np.random.random((4, 5))
print(g)

#3から10まで1ずつ増加する数列を作成(10は含めない)
h = np.arange(3, 10, 1)
print(h)

val = g[0, 1]
print(val)

center = g[1:3, 1:4]
print(center)

print("shape of g:", g.shape)
print("shape of center:", center.shape)

print(g)
g[1:3, 1:4] = 0

print(g)

i = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)

print(i)

print(np.array([i[0, 1], i[2, 1], i[1, 0]]))
print(i[[0, 2, 1], [1, 1, 0]])

j = np.array([1, 2, 3])

print("type:", j.dtype)

k = np.array([1., 2., 3.])

print("type:", k.dtype)

l = np.array([1, 2, 3], dtype=np.float32)

print("type:", l.dtype)

m = np.array([1, 2, 3], dtype='f')

print("type:", m.dtype)

#データ型の変換
n = m.astype(np.float64)

print("type:", n.dtype)

#配列の計算
a = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

b = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

c = a + b

print(c)

#要素ごとに平方根を計算
c = np.sqrt(b)
print(c)

#要素ごとにn乗
n = 2
c = np.power(b, n)
print(c)
print(c ** n)

a = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

b = np.array([1, 2, 3])

c = a + b

print(c)


a = np.array([1, 2, 3])
print(a)

#どちらも同じことをやっている
b = np.array([2, 2, 2])
c = a * b
print(c)

c = a*2
print(c)


#ブロードキャストの例
a = np.random.randint(0, 10, (2, 1, 3))
b = np.random.randint(0, 10, (3, 1))

print('a:\n', a)
print('\na.shape:', a.shape)
print('\nb:\n', b)
print('\nb.shape:', b.shape)

c = a + b
print('\na + b:\n', c)
print('\n(a + b).shape:', c.shape)


print('Original shape:', b.shape)

b_expanded = b[np.newaxis, :, :]
print('Added new axis to the top:', b_expanded.shape)

b_expanded2 = b[:, np.newaxis, :]
print('Added new axis to the middle:', b_expanded2.shape)

print(b)

print(b_expanded)

print(b_expanded2)

#実行時間の比較
import time

a = np.array([
    [0, 1, 2, 1, 0],
    [3, 4, 5, 4, 3],
    [6, 7, 8, 7, 6],
    [3, 4, 5, 4, 4],
    [0, 1, 2, 1, 0]
])

b = np.array([1, 2, 3, 4, 5])
c = np.empty((5, 5))

start_time = time.perf_counter()
for i in range(a.shape[0]):
    c[i, :] = a[i, :] + b
end_time = time.perf_counter()
print("time:", end_time - start_time)
print(c)

#ブロードキャストを使ったほうが短い
start_time = time.perf_counter()
c = a + b
end_time = time.perf_counter()
print("time:", end_time - start_time)
print(c)


A = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

C = np.dot(A, B)
print(C)

C = A.dot(B)
print(C)

print(a.dtype)

x = np.random.randint(0, 10, (8, 10))
print(x)

#平均値
print(x.mean())

#分散
print(x.var())

#標準偏差
print(x.std())

#最大値
print(x.max())

#最小値
print(x.min())

#axis次元に沿って平均を求める
print(x.mean(axis=1))

#上と同じ
print(
    np.array([
    x[0, :].mean(),
    x[1, :].mean(),
    x[2, :].mean(),
    x[3, :].mean(),
    x[4, :].mean(),
    x[5, :].mean(),
    x[6, :].mean(),
    x[7, :].mean(),
])
)

X = np.array([
    [2, 3],
    [2, 5],
    [3, 4],
    [5, 9],
])
print(X)

#Xのデータ数と同じ数だけ1が並んだ配列
ones = np.ones((X.shape[0], 1))

#1次元目に1を付け加える
X = np.concatenate((ones, X), axis=1)

print(X)

t = np.array([1, 5, 6, 8])
print(t)


#正規方程式の解
# w = (X^T X)^-1 X^T tの計算を行う

#X^T Xの計算  .Tは転置
xx = np.dot(X.T, X)
print(xx)

#np.linalg.inv()は逆行列
xx_inv = np.linalg.inv(xx)
print(xx_inv)

#X^T tの計算
xt = np.dot(X.T, t)
print(xt)

w = np.dot(xx_inv, xt)
print(w)

#上記と同じ計算
w_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)
print(w_)

w_ = np.linalg.solve(X.T.dot(X), X.T.dot(t))
print(w_)