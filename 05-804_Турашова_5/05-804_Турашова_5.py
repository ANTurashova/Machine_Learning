"""
5. Класс cmeans
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class C_Means():
    def __init__(self, dataset, n_clusters=3, fuzzy=3, cut=0.8):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.fuzzy = fuzzy  # Параметр размытости
        self.cut = cut  # Параметр отсечения
        self.max_n_iter = 100
        self.tolerance = 0.001
        self.fitted = False
        self.labels = np.array([], dtype='i')
        self.centroids = self.dataset[np.random.choice(self.dataset.shape[0], size=self.n_clusters, replace=False)]
        self.U = np.array([], dtype='f')  # Матрица принадлежности

    def get_dist(self, list1, list2):
        """
        Метод для измерения расстояния между двумя объектами
        """
        return np.sqrt(sum((i-j)**2 for i, j in zip(list1, list2)))

    def distribute_data(self):
        """
        Метод пересчета матрицы принадлежности U
        """
        dist = np.array([[self.get_dist(center, data) for center in self.centroids] for data in self.dataset])
        for k, d in enumerate(dist):
            for kk, dd in enumerate(d):
                if dd == 0:
                    dist[k][kk] = 0.0001
        self.U = (1 / dist) ** self.fuzzy
        self.U = self.U / np.array([self.U.sum(axis=1)]).T

    def get_labels(self):
        """
        Метод заполнения массива меток (labels) обучающей выборки
        с использованием параметра отсечения (cut)
        """
        self.labels = np.array([], dtype='i')
        for u in self.U:
            if max(u) > self.cut:
                self.labels = np.append(self.labels, list(u).index(max(u)) + 1)
            else:
                self.labels = np.append(self.labels, 0)

    def recalculate_centroids(self):
        """
        Метод пересчета центров кластеров с помощью матрицы принадлежности
        """
        for j in range(self.n_clusters):
            sum_1 = 0
            for i, u_i in enumerate(self.U):
                sum_1 += u_i[j] * self.dataset[i]  # sum(uj_i * x_i)
            sum_2 = self.U.sum(axis=0)[j]  # sum(uj_i)
            self.centroids[j] = sum_1/sum_2  # cj = sum(uj_i * x_i) / sum(uj_i)

    def fit(self):
        """
        Метод обучения модели
        """
        self.distribute_data()
        self.get_labels()
        iter = 1
        while iter < self.max_n_iter:
            prev_centroids = self.centroids.copy()
            self.recalculate_centroids()
            self.distribute_data()
            self.get_labels()
            if np.array([self.get_dist(i, j) for i, j in zip(prev_centroids, self.centroids)]).max() < self.tolerance:
                print("iter:", iter)
                break
            iter += 1
            # plt.scatter(X[:, 0], X[:, 1], c=self.labels, cmap=plt.cm.Paired)
            # plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', marker='+')
            # plt.show()
        self.fitted = True

    def predict(self, list2d):
        """
        Метод предсказания меток для массива новых объектов
        """
        if self.fitted:
            print(list2d)
            list2d = np.array(list2d)

            dist = np.array([[self.get_dist(center, data) for center in self.centroids] for data in list2d])
            for k, d in enumerate(dist):
                for kk, dd in enumerate(d):
                    if dd == 0:
                        dist[k][kk] = 0.0001
            U = (1 / dist) ** self.fuzzy
            U = U / np.array([U.sum(axis=1)]).T

            labels = np.array([], dtype='i')
            for u in U:
                if max(u) > self.cut:
                    labels = np.append(labels, list(u).index(max(u)) + 1)
                else:
                    labels = np.append(labels, 0)
            print(labels)


X, _ = make_blobs(n_samples=300, random_state=16)
cmeans = C_Means(X, n_clusters=3, fuzzy=3, cut=0.8)
cmeans.fit()

plt.scatter(X[:, 0], X[:, 1], c=cmeans.labels, cmap=plt.cm.Paired)
plt.scatter(cmeans.centroids[:, 0], cmeans.centroids[:, 1], c='black', marker='+')
plt.show()

cmeans.predict([[-5, 0], [-2, -6], [0, -8], [2, -10], [10, -20]])
