#Sepal length, Sepal width, Petal length, Petal widthの4つのデータからアヤメの種類(3種類)を推定する
#https://tutorials.chainer.org/ja/14_Basics_of_Chainer.html

from keras.metrics import top_k_categorical_accuracy
from sklearn import model_selection
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

x, t = load_iris(return_X_y=True)

#print("x:", x.shape)
print("t:", t.shape)

#データの標準化
x = preprocessing.scale(x)

#one hot encodingに変換
t = to_categorical(t)
print("t:", t.shape)


#データセットの分割
from sklearn.model_selection import train_test_split


#_testはテスト用, _trainを用いて訓練
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
import numpy as np

n_in = 4
n_out = 3
hidden_units = 10
hidden_layers = 1
learning_late = 0.01


model = Sequential()

model.add(Dense(n_in, activation = "relu"))

for i in range(hidden_layers):
    model.add(Dense(hidden_units, activation = "relu"))


model.add(Dense(n_out, activation="softmax"))

model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = learning_late), metrics=["accuracy"])

n_epoch = 30
n_batchsize = 16


history = model.fit(x_train, t_train, epochs = n_epoch, batch_size = n_batchsize, validation_split=0.3)

import matplotlib.pyplot as plt

#描画
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


print("Prediction///////////////")
predict = model.predict(x_test)
P = np.empty(predict.shape)
for i in predict:
    if i[0] > i[1] and i[0] > i[2]:
        np.append(P, [[1,0,0]])
    elif i[1] > i[0] and i[1] > i[2]:
        np.append(P, [[0,1,1]])
    else:
        np.append(P, [[0, 0, 1]])


print(x_test)#データ
print(predict)#予想結果
print(predict.shape)
print(t_test)#答え
print(P)

#print(predict[1])
P = to_categorical(predict)
print(P.shape)
#print(t_test[1])
#if P[0] == t_test[0]:
#    print("合致")



