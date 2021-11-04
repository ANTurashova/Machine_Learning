import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

X = np.array([[1, 1], [1, 2], [2, 1], [7, 8], [6, 7]])
print(X)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
print("-------------------------")

X = np.random.rand(100, 2)
plt.scatter(X[:, 0], X[:, 1], c='r', marker='+')  # c='r' цвет красный
plt.show()
print("-------------------------")

# X = make_blobs(n_samples=300)  # метки к каким группам относятся точки
# print(X)
X, _ = make_blobs(n_samples=300, random_state=42)  # random_state=42 специальная генирация
plt.scatter(X[:, 0], X[:, 1], c='r', marker='+')
plt.show()
print("-------------------------")

X, _ = make_circles(n_samples=300, noise=.05, factor=.4)  # noise=.05 насколько точки далеко от окружностей.
# factor=.4 насколько окружности далеко друг от друга.
plt.scatter(X[:, 0], X[:, 1], c='r', marker='+')
plt.show()
print("-------------------------")

X = load_iris()['data']  # load_iris словарь, вытаскиваем всё по ключу data
print(X)
plt.scatter(X[:, 0], X[:, 3], c='r', marker='+')
plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2])
# plt.show()
print("-------------------------")

X, _ = make_blobs(n_samples=300, random_state=42)  # random_state=42 специальная генирация по шаблону 42
kmeans = KMeans(n_clusters=3)  # n_clusters=3 количество кластеров
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_  # центр группы
print(labels)
print(centroids)
color_list = np.array(['red', 'yellow', 'green', 'blue', 'pink', 'black', '#f2890a'])
print(color_list[[1, 4, 0, 1, 2]])
plt.scatter(X[:, 0], X[:, 1], c=color_list[labels])
plt.scatter(centroids[:, 0], centroids[:, 1], c='b', marker='+')
plt.show()
print(kmeans.predict([[-6, 10], [5, -2]]))  # предсказание того, в какой кластер новые точки будут попадать
print("-------------------------")

inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, 8), inertia)
plt.show()  # график, который показывать оптимальное количество кластеров. Где излом там и цифра внизу подскажет
print("-------------------------")

X, _ = make_circles(n_samples=300, noise=.04, factor=.5)
kmeans = KMeans(n_clusters=3).fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_  # центр группы
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Paired)  # cmap=plt.cm.Paired палитра
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='+')
plt.show()

