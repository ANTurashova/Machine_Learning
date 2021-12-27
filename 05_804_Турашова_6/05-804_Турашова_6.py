"""
6. Подбор гиперпараметров метода DB_SCAN

Дано:
- дан класс DB_SCAN кластеризации данных

Требуется:
- реализовать эвристический метод побора гиперпараметра eps:

1. Выберите значение гиперпараметра m. Обычно используются значения от 3 до 9, чем более неоднородный ожидается
датасет, и чем больше уровень шума, тем большим следует взять m.

2. Вычислите среднее расстояние по m ближайшим соседям для каждой точки. Т.е. если m=3, нужно выбрать трёх ближайших
соседей, сложить расстояния до них и поделить на три.

3. Отсортируйте полученные значения по возрастанию и выведите на экран.

4. Гиперпараметр eps следует взять в полосе, где происходит самый сильный перегиб графика. Чем больше eps, тем больше
получатся кластеры, и тем меньше их будет.

- проверить результат подбора параметра eps - показать результат кластеризации для заданного распределение точек при
eps меньшим, равным и большим рекомендованного значения
- (не обязательно) попытаться автоматизировать подбор eps согласно эвристики без участия человека
(без построения и анализа графика)
"""

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
import importlib
import matplotlib.pyplot as plt
import numpy as np
# from 05_804_Турашова_6.create_points import main
# Импорт модуля, имя котороего не проходит стандарты:
create_points = importlib.import_module('05_804_Турашова_6.create_points')


points = create_points.main()
fig = plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], c="black")
plt.show()
fig.savefig('initial_drawing.png')


# 1. Выберите значение гиперпараметра m. Обычно используются значения от 3 до 9, чем более неоднородный ожидается
# датасет, и чем больше уровень шума, тем большим следует взять m.
m = 3


# 2. Вычислите среднее расстояние по m ближайшим соседям для каждой точки. Т.е. если m=3, нужно выбрать трёх ближайших
# соседей, сложить расстояния до них и поделить на три.
d = pairwise_distances(points)
d[d == 0] = np.inf
y = np.sort(d, axis=1)[:, :m].mean(axis=1)


# 3. Отсортируйте полученные значения по возрастанию и выведите на экран.
data_new = sorted(y)
plt.plot(data_new)
plt.savefig('schedule.png')


# 4. Гиперпараметр eps следует взять в полосе, где происходит самый сильный перегиб графика. Чем больше eps, тем больше
# получатся кластеры, и тем меньше их будет.
# Возьмём три значения eps:
labels = DBSCAN(eps=5, min_samples=2).fit_predict(points)
fig = plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Paired)
plt.show()
fig.savefig('eps_5.png')

labels = DBSCAN(eps=10, min_samples=2).fit_predict(points)
fig = plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Paired)
plt.show()
fig.savefig('eps_10.png')

labels = DBSCAN(eps=15, min_samples=2).fit_predict(points)
fig = plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Paired)
plt.show()
fig.savefig('eps_15.png')
