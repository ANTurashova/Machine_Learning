"""
8. Искусственные нейронные сети

Дано:
- код ИНС прямого распространения с двумя скрытыми слоями

##################################################

def sigmoid(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

eta = 1
epochs = 10000

W0 = np.random.rand(2+1,5)
W1 = np.random.rand(5+1,4)
W2 = np.random.rand(4+1,3)
size = len(X_)
bias = np.ones((size,1))

for epoch in range(epochs):
    # прямое распространение информации
    L0 = np.append(X_, bias, axis=1)
    Z1 = sigmoid(np.dot(L0,W0))
    L1 = np.append(Z1, bias, axis=1)
    Z2 = sigmoid(np.dot(L1,W1))
    L2 = np.append(Z2, bias, axis=1)
    Z3 = sigmoid(np.dot(L2,W2))
    L3 = Z3
    # обратное распространение ошибки
    delta3 = y_vect - Z3
    delta2 = (np.dot(delta3*sigmoid(Z3,True),W2.T))[:,:-1]
    delta1 = (np.dot(delta2*sigmoid(Z2,True),W1.T))[:,:-1]
    # обновление весов
    W2 += eta*np.dot(L2.T,delta3*sigmoid(Z3,True))/size
    W1 += eta*np.dot(L1.T,delta2*sigmoid(Z2,True))/size
    W0 += eta*np.dot(L0.T,delta1*sigmoid(Z1,True))/size

def predict(X):
    bias = np.ones((len(X),1))
    X_ = X_scale.transform(X)
    L0 = np.append(X_, bias, axis=1)
    Z1 = sigmoid(np.dot(L0,W0))
    L1 = np.append(Z1, bias, axis=1)
    Z2 = sigmoid(np.dot(L1,W1))
    L2 = np.append(Z2, bias, axis=1)
    Z3 = sigmoid(np.dot(L2,W2))
    L3 = Z3
    return np.argmax(Z3, axis=1)

##################################################


Требуется:
- Изменить функцию активации на гиперболический тангенс;
- Добавить возможность создания сети с произвольным количеством скрытых слоев и нейронах на этих слоях.
Размерность обучающих данных определит количество нейронов во входном слое (X.shape[1]),
переменная num_class задает количество нейронов на выходном слое.
Список num_neurons_hid_layers задает количество скрытых слоев и нейронов на них.

Например:

num_class = 3 # количество классов на выходе ИНС
num_neurons_hid_layers = [10,8,6] # три скрытых слоя с 10, 8 и 6 нейронами соответственно
# если X.shape[1] = 2 (два признака у каждого объекта),
# то программа для работы сети определит следующие матрицы весов (с учетом нейронов сдвига)
# W[0] = np.random.random((2+1,10))
# W[1] = np.random.random((10+1,8))
# W[2] = np.random.random((8+1,6))
# W[3] = np.random.random((6+1,3))
"""
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import importlib
# from 05_804_Турашова_8.nnet import NNeT
# Импорт модуля, имя котороего не проходит стандарты:
nnet = importlib.import_module('05_804_Турашова_8.nnet')


def convert_y_to_vect(y, num_class=2):
    y_vect = np.zeros((len(y), num_class))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect


# Подготавливаем данные:
x_1, x_2 = np.meshgrid(np.arange(-5, 5), np.arange(-5, 5))
x = np.array([x_1.flatten(), x_2.flatten()]).T
y = [0] * x.shape[0]
x_scale = StandardScaler()
x_fit = x_scale.fit_transform(x)
for i in range(x.shape[0]):
    if np.sin(x[i, 0]) > x[i, 1]:
        y[i] = 1
    if np.cos(x[i, 0])-4 > x[i, 1]:
        y[i] = 2
y_vect = convert_y_to_vect(y, 3)
fig = plt.figure()
plt.scatter(x_fit[:, 0], x_fit[:, 1], c=y, cmap=plt.cm.Paired)
plt.show()
fig.savefig('training.png')


# Обучение:
model = nnet.NNet(x_fit.shape[1], [6, 8, 5], 3)
model.train(x_fit, y_vect, activation="tanh", epochs=1000, eta=1)


# Подготавливаем данные для предсказания:
x_1, x_2 = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
X_test = np.array([x_1.flatten(), x_2.flatten()]).T


# Предсказываем:
x_scale = StandardScaler()
X_test_scaled = x_scale.fit_transform(X_test)
pred = model.predict(X_test_scaled)
fig = plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=pred, cmap=plt.cm.Paired)
plt.show()
fig.savefig('prediction.png')
