"""
4. Класс kmeans

Дано:
- дан класс K_Means кластеризации данных

Требуется:
- изменить способ задания начальных положений центров кластеров: разбросать случайным образом между(!) точек
обучающего множества dataset (определив разброс точек по всем координатам)
- изменить способ задания параметра tolerance: значение параметра должно равняться 1/10000 от минимального
диапазона изменения признаков объектов (например, если один признак изменятся в диапазоне [100,200], а другой -
[0.1,10], то tolerance = (10-0.1)/10000)
- дописать метод get_dist: пользователь должен иметь возможность использовать одну из 4 метрик при подсчете
расстояний между объектами (евклидово расстояние, квадрат евклидова расстояния, расстояние городских кварталов,
расстояние Чебышёва)
- написать метод get_labels: методом можно воспользоваться только после обучения модели (fitted == True), метод
принимает двумерный массив объектов (могут быть точки не принадлежащие исходному обучающему множеству dataset),
метод возвращает одномерный массив индексов соответствующих кластеров (по близости до центров кластеров)
- назвать файл по шаблону на латинице без пробелов: Группа_Фамилия_НомерРаботы (например, 11-000_Ivanov_4)
"""


import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import make_blobs


class K_Means():
    def __init__(self, dataset, n_clusters=3, metric_number=0):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.metric_number = metric_number
        self.centroids = self.random_centroids_and_tolerance()[0]
        print("self.centroids:", self.centroids)
        self.labels = np.array([], dtype='i')
        self.fitted = False
        self.max_n_iter = 100
        self.tolerance = self.random_centroids_and_tolerance()[1]
        print("self.tolerance:", self.tolerance)

    """Подсчёт centroids и tolerance"""
    def random_centroids_and_tolerance(self):
        x_max = -10000
        x_min = 10000
        y_max = -10000
        y_min = 10000
        centroids = np.array([], dtype='f')
        for i in self.dataset:
            if x_max < i[0]:
                x_max = i[0]
            elif x_min > i[0]:
                x_min = i[0]
            if y_max < i[1]:
                y_max = i[1]
            elif y_min > i[1]:
                y_min = i[1]
        for i in range(self.n_clusters):
            centroids = np.append(centroids, random.uniform(x_min, x_max))
            centroids = np.append(centroids, random.uniform(y_min, y_max))
        centroids = centroids.reshape(self.n_clusters, 2)
        if (x_max - x_min) < (y_max - y_min):
            tolerance = (x_max - x_min) / 10000
        else:
            tolerance = (y_max - y_min) / 10000
        return centroids, tolerance

    """
    Нахождение расстояния между точками.
    Eвклидово расстояние: self.metric_number = 0
    Квадрат евклидова расстояния: self.metric_number = 1
    Расстояние городских кварталов: self.metric_number = 2
    Расстояние Чебышёва: self.metric_number = 3
    """
    def get_dist(self, list1, list2):
        metric = 0
        if self.metric_number == 0:
            metric = np.sqrt(sum((i-j)**2 for i, j in zip(list1, list2)))
        if self.metric_number == 1:
            metric = sum((i-j)**2 for i, j in zip(list1, list2))
        if self.metric_number == 2:
            metric = sum(abs(i-j) for i, j in zip(list1, list2))
        if self.metric_number == 3:
            metric = max(abs(i-j) for i, j in zip(list1, list2))
        return metric

    """Распределение точек по имеющимся класстерам"""
    def distribute_data(self):
        self.labels = np.array([], dtype='i')
        for data in self.dataset:
            self.labels = np.append(self.labels,
                                    np.array([self.get_dist(data, center) for center in self.centroids]).argmin())

    """Пересчёт цетров класстеров"""
    def recalculate_centroids(self):
        for i in range(self.n_clusters):
            num = 0
            temp = np.zeros(self.dataset[0].shape)
            for k, label in enumerate(self.labels):
                print("k, label", k, label)
                if label == i:
                    print("self.dataset[k]", self.dataset[k])
                    temp += self.dataset[k]
                    print("temp", temp)
                    num += 1
            self.centroids[i] = temp/num

    """Нахождение правильного распределения центров кластеров"""
    def fit(self):
        iter = 0
        while iter < self.max_n_iter:
            prev_centroids = self.centroids.copy()
            self.distribute_data()
            self.recalculate_centroids()
            if np.array([self.get_dist(i, j) for i, j in zip(prev_centroids, self.centroids)]).max() < self.tolerance:
                print("iter:", iter)
                break
            iter += 1
        self.fitted = True

    """Проверка, к каким класстерами принадлежат введённые пользователем точки"""
    def get_labels(self, list2d):
        if self.fitted:
            print("list2d:", list2d)
            list2d = np.array(list2d)
            labels = np.array([], dtype='i')
            for data in list2d:
                labels = np.append(labels, np.array([self.get_dist(data, center) for center in self.centroids]).argmin())
            return labels


X, _ = make_blobs(n_samples=300, random_state=2)
kmeans = K_Means(X, n_clusters=3, metric_number=0)
kmeans.distribute_data()
kmeans.recalculate_centroids()
kmeans.fit()

print("kmeans.get_labels:", kmeans.get_labels([[3, 2], [0, -4], [-3, -10], [-4, -12]]))

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='black', marker='+')
plt.show()
