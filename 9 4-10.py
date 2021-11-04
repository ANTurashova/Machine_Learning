from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np

# points, _ = make_circles(n_samples=300, noise=.04, factor=.5)
points, _ = make_circles(n_samples=300, noise=.08, factor=.5)


print("---------")
labels = DBSCAN(eps=.2, min_samples=8).fit_predict(points)
print(labels)
plt.figure()
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.Paired)
plt.show()
