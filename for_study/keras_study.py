#Sepal length, Sepal width, Petal length, Petal widthの4つのデータからアヤメの種類(3種類)を推定する
#https://tutorials.chainer.org/ja/14_Basics_of_Chainer.html

from keras.metrics import top_k_categorical_accuracy
from sklearn import model_selection
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from tensorflow.python.keras.layers.core import Dropout

x, t = load_iris(return_X_y=True)

#print("x:", x.shape)
#print("t:", t.shape)

#データの標準化
x = preprocessing.scale(x)

#one hot encodingに変換
print(t)
print(t.shape)
t = to_categorical(t)
print("t:", t.shape)
print(t)


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
hidden_units = 16
hidden_layers = 4
learning_late = 0.05


model = Sequential()

model.add(Dense(hidden_units, input_dim=n_in, activation="relu"))

for i in range(hidden_layers):
    model.add(Dense(hidden_units, activation = "relu"))

model.add(Dense(n_out, activation="softmax"))
#多クラス分類はsoftmaxとcategorical crossentropyがよく使われる
model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = learning_late), metrics=["accuracy"])

n_epoch = 48
n_batchsize = 32


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


#予測
print("Prediction")
predict = model.predict(x_test)

j=0
k=0
for i, test in enumerate(predict):

    if (t_test[i] == np.round(test)).all():
        j+=1
    k+=1

print("正解率:",round(100 * j / x_test.shape[0], 1), "%")
#だいたい80%~97%



