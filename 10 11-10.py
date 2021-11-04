import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# points, _ = make_circles(n_samples=300, noise=.05, factor=.4)
points, _ = make_blobs(n_samples=300, random_state=42)
plt.scatter(points[:, 0], points[:, 1], c='r', marker='+')
labels = DBSCAN(eps=20, min_samples=15).fit_predict(points)
plt.figure()
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Paired)
plt.show()

class DB_SCAN():
    def __init__(self, dataset, eps=20.0, min_samples=10):
        self.dataset = dataset
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = 0
        self.clusters = {0: []}
        self.visited = set()
        self.clustered = set()
        self.labels = np.array([], dtype='i')
        self.fitted = False


    def get_dist(self, list1, list2):
        return np.sqrt(sum((i - j) ** 2 for i, j in zip(list1, list2)))


    def get_neighbours(self, P):
        return [list(Q) for Q in self.dataset if self.get_dist(P, Q) < self.eps]


    def expand_cluster(self, P):
        self.n_clusters += 1
        self.clusters[self.n_clusters] = []
        self.clustered.add(tuple(P))
        self.clusters[self.n_clusters].append(P)
        neighbours = self.get_neighbours(P)
        while neighbours:
            Q = neighbours.pop()
            if tuple(Q) not in self.visited:
                self.visited.add(tuple(Q))
                Q_neighbours = self.get_neighbours(Q)
                if len(Q_neighbours) > self.min_samples:
                    neighbours.extend(Q_neighbours)
            if tuple(Q) not in self.clustered:
                self.clustered.add(tuple(Q))
                self.clusters[self.n_clusters].append(Q)
                if Q in self.clusters[0]:  # python list required not np.array
                    self.clusters[0].remove(Q)


    def fit(self):
        for P in self.dataset:
            P = list(P)
            if tuple(P) in self.visited:
                continue
            self.visited.add(tuple(P))
            neighbours = self.get_neighbours(P)
            if len(neighbours) < self.min_samples:
                self.clusters[0].append(P)
            else:
                self.expand_cluster(P)
        self.fitted = True


    def get_labels(self):
        labels = []
        if not self.fitted:
            self.fit()
        for P in self.dataset:
            for i in range(self.n_clusters + 1):
                if list(P) in self.clusters[i]:
                    labels.append(int(i))
        self.labels = np.array(labels, dtype='i')
        return self.labels


dbscan = DB_SCAN(points, eps=20., min_samples=10)
print(dbscan.labels)
print(dbscan.get_neighbours(dbscan.dataset[39]))

# dbscan.expand_cluster(dbscan.dataset[39])
# print(dbscan.clusters)

dbscan.fit()
# Из-за этой строки были проблемы
# print(dbscan.clusters)

dbscan.get_labels()
print(dbscan.labels)

#----

labels = DB_SCAN(points, eps=20., min_samples=10).get_labels()

fig = plt.figure()
plt.scatter(points[:,0], points[:,1], c=labels, cmap=plt.cm.Paired)
plt.show()


#-------


l = [2.5, 8.9]
t = tuple(l)
n = np.array(l)
t = tuple(n)
print(t)
s = set()
s.add(t)
s.add(t)
s