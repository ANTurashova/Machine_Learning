import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

orig_img = imread('img/butterfly.jpg')
plt.imshow(orig_img)
plt.show()
print(orig_img.shape)
print(orig_img)
print("-------------------------")

X = orig_img.reshape((-1, orig_img.shape[-1]))  # перегруппируем пиклеси изображения в одну строку
print(X)
n_color = 8

kmeans = KMeans(n_clusters=n_color).fit(X)
# kmeans = KMeans(n_clusters=n_color).fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap=plt.cm.Paired)
plt.show()
print("-------------------------")

new_img = centroids[labels].astype('int').reshape(orig_img.shape)  # reshape(orig_img.shape) изменение разметрности обратно к оригинальной картинке
print(new_img)
plt.imshow(new_img)
plt.show()


