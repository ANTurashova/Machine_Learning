import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class K_Means():
    def __init__(self, dataset, n_clusters=3):
        self.dataset = dataset
        self.n_clusters = n_clusters
        self.centroids = np.array([self.dataset[k] for k in range(self.n_clusters)], dtype='f')  # потом задать случайно
        # по каждому признаку сделать диапазон
        self.labels = np.array([], dtype='i')  # индекс класстера, к котору относится соотв точка
        self.fitted = False  # флаг - обучена ли модель?
        self.max_n_iter = 100
        self.tolerance = .01  # тут константа, показывающая мин смещение центров класстеров для завершения обучения

    def get_dist(self, list1, list2):  # зададим метрику Евклида, покажет расстояние между точками
        return np.sqrt(sum((i-j)**2 for i, j in zip(list1, list2)))
        # zip извлекает последовательно пары элементов и помещает их в i,j

    def distribute_data(self):  # распределение точек по имеющимся класстерам
        self.labels = np.array([], dtype='i')
        # for data in self.dataset:  # расстояние от центра первого класстера до всех точек
        #     print(self.get_dist(data, self.centroids[0]))
        # for data in self.dataset:  # расстояние от центров класстеров до точек
        #     print([self.get_dist(data, center) for center in self.centroids])
        for data in self.dataset:  # номер столбца, в котором находится минимальное расстрояние
            # print(np.array([self.get_dist(data, center) for center in self.centroids]).argmin())
            # для каждой точки вывели индекс класстера, расстояние до которого минимально
            self.labels = np.append(self.labels,
                                    np.array([self.get_dist(data, center) for center in self.centroids]).argmin())

    def recalculate_centroids(self):  # пересчёт цетров класстеров
        for i in range(self.n_clusters):
            num = 0
            temp = np.zeros(self.dataset[0].shape)
            for k, label in enumerate(self.labels):  # enumerate извлекает не только метки, но и ещё k
                if label == i:
                    temp += self.dataset[k]
                    num += 1  # если такая точка нашлась, то счётчик увеличиваем
            self.centroids[i] = temp/num  # после того, как пробежались по всем точкам, находим усреднение

    def fit(self):  # повторять распределение данных и пересчёт центров много раз
        iter = 0
        while iter < self.max_n_iter:
            prev_centroids = self.centroids.copy()  # зафикс прерыдущее положение центроидов
            self.distribute_data()
            self.recalculate_centroids()
            if np.array([self.get_dist(i, j) for i, j in zip(prev_centroids, self.centroids)]).max() < self.tolerance:
                # находим мааксимальное расстояние между центрами сентроидов, если оно меньше заданного нами, то
                print(iter)
                break
            iter += 1
        self.fitted = True

    def predict(self, list2d):
        return #  list1d --> [0, 2, 1...] это лист меток. Должен вернуть лист меток.


X, _ = make_blobs(n_samples=300, random_state=2)
kmeans = K_Means(X, n_clusters=3)
print(kmeans.labels)
print(kmeans.centroids)
print(kmeans.get_dist([0, 0], [1,1]))

kmeans.distribute_data()
print(kmeans.labels)

kmeans.recalculate_centroids()
print(kmeans.centroids)

kmeans.fit()

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap=plt.cm.Paired)  # cmap=plt.cm.Paired палитра
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='black', marker='+')
plt.show()


