import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave

x = 2.654
print(x**3)
print(np.sqrt(x))
print(np.log(x))
print(np.exp(x))
print("-------------------------")

print(np.random.randint(30, 100))
print(np.random.uniform(30, 100))
print(np.random.rand(3, 4))
print("-------------------------")

print(np.zeros((2, 5)))
print(np.ones((6, 3), dtype="i"))
print(np.linspace(3, 14, 11))
print("-------------------------")

list1 = [3, 6, 2, 4]
np_list = np.array(list1)
print(list1)
print(np_list)
print(list1*3)
print(np_list*3)
list2 = [[1, 2, 3], [4, 5, 6]]
np_list2 = np.array(list2)
print(list2)
print(np_list2)
print(np_list2.shape)
print(np_list2.reshape(3, 2))
print(np_list2.reshape(1, 6))
print("-------------------------")

A = np.random.rand(2, 4)
B = np.ones((4, 3))
C = 2 * B
B[1, 1] = 2
print(A)
print(B)
print(C)
print("-------------------------")

print(B + C)
print(B - C)
print(B * C)
print(np.dot(A, B))
print(np.dot(B.T, C))
print("-------------------------")

Y = np.random.rand(10, 5)
print(Y)
print("-------------------------")

print(Y[4, 2])
print(Y[:, 2])
print(Y.min(axis=0))  # минимальные числа в каждом из столбцов
print(Y.min(axis=1))  # минимальные числа в каждом из строк
print(Y.argmin(axis=1))
print(Y[1:4, 2:5])
print(Y[1:4, 2:5].argmin(axis=1))
print("-------------------------")

plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
plt.scatter([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], c="red", marker="+")
plt.show()
print("-------------------------")

x = np.linspace(1, 15, 20)
y = np.log(x)
plt.plot(x, y)
plt.show()
print("-------------------------")

pic = np.random.rand(2, 5)
print(pic)
fig, ax = plt.subplots()
ax.imshow(pic)
plt.show()
print("-------------------------")

orig_img = imread('img/kitty.jpg')
plt.imshow(orig_img)
plt.show()
print(orig_img.shape)
print(orig_img)
print("-------------------------")

img_fliph = orig_img[:, ::-1, :].copy()
plt.imshow(img_fliph)
plt.show()
print("-------------------------")

test = orig_img.copy()
test[100:250, 400:550, 0] = 255
test[100:250, 400:550, 2] = 255
test[100:130, 400:430, 0] = 0
plt.imshow(test)
plt.show()
imsave('data/01_new.jpg', test)
