import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# В c_means главное составление матрицы принадлежности

fuzzy = 2

dataset, _ = make_blobs(n_samples=300, random_state=2)

n_clusters = 3

centroids = dataset[np.random.choice(dataset.shape[0], size=n_clusters)]


def get_dist(list1, list2):  # зададим метрику Евклида, покажет расстояние между точками
    return np.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))
    # zip извлекает последовательно пары элементов и помещает их в i,j


dist = np.array([[get_dist(center, data) for center in centroids] for data in dataset])
# print(dist)

U = (1/dist)**fuzzy
# print(U)

print(U / np.array([U.sum(axis=1)]).T)

print((U / np.array([U.sum(axis=1)]).T).sum(axis=1))  # sum(axis=1) сумма элементов в строке
print((U / U.sum(axis=1)[:, None]).sum(axis=1))

