"""
3. Суперпиксельная сегментация изображения

Дано:
- изображение (формат jpg/png)
- количество кластеров N
- задан нормировочный коэффициент alpha

Требуется:
- каждый пиксель представить в виде объекта с набором признаков {R, G, B, alpha*X, alpha*Y} (где R – количество
красного цвета, G – зеленого, B – синего, X - номер столбца, Y - номер строки)
- провести кластеризацию пикселей методом kmeans
- заменить в исходном изображении цвет каждого пикселя на цвет центра кластера, которому принадлежит пиксель
- повторить кластеризацию для разных значений коэффициента alpha
- прислать файл(ы): исходное изображение, изображения после кластеризации для нескольких (больших и малых)
значений коэффициента alpha, код (для работы в формате ipynb - всё в одном файле, в противном случае - все файлы
поместить в один архив)
- назвать файл по шаблону на латинице без пробелов: Группа_Фамилия_НомерРаботы (например, 11-000_Ivanov_3)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread, imsave
from sklearn.cluster import KMeans


# orig_img = imread('kitty.jpg')
orig_img = imread('butterfly.jpg')
# orig_img = imread('docker.png')
plt.imshow(orig_img)
plt.show()
N = 36
alpha = 1


print("Добавляем координаты х и у в массив пикселей и получаем формат [r g b x y r g b x y ...]")
schet_x = 0
schet_y = 0
img_5 = np.array([], dtype=np.int)
for i in orig_img:
    schet_x = 0
    for ii in i:
        ii = np.append(ii, schet_x * alpha)
        ii = np.append(ii, schet_y * alpha)
        img_5 = np.append(img_5, ii)
        schet_x += 1
    schet_y += 1
print(img_5)
print("------------------------------")


print("Преобразуем массив в формат [[r g b x y] [r g b x y] ...]")
X = img_5.reshape(orig_img.shape[0] * orig_img.shape[1], orig_img.shape[2] + 2)
print(X)
print("------------------------------")


print("Выделяем кластеры")
kmeans = KMeans(n_clusters=N).fit(X)
labels = kmeans.labels_  # Список с циферками, обозначающими принадлежность к классу
centroids = kmeans.cluster_centers_  # лист листов с rgb кластеров
if orig_img.shape[2] == 3:  # Если jpg, то преобразуем в int
    centroids = centroids.astype('int')
print("------------------------------")


print("Переприсваиваем цвета")
new_img_5 = centroids[labels]  # reshape(orig_img.shape) изменение разметрности обратно к оригинальной картинке
print(new_img_5)
print("------------------------------")


print("Удаляем x y из массива и преобразуем к виду [[[r g b] [r g b] ...] ...]]")
new_img = np.array([], dtype=np.int)
for i in new_img_5:
    i = np.delete(i, i.shape[0] - 1)
    i = np.delete(i, i.shape[0] - 1)
    new_img = np.append(new_img, i)
new_img = new_img.reshape(orig_img.shape)
print(new_img)
print("------------------------------")


plt.imshow(new_img)
plt.show()


print("Если изображение jpg, то мы его преобразуем в float с диапозоном 0.0-1.0 и сохраним")
print("У png проблемы с преобразованием в диапозон 0.0 - 1.0")
if orig_img.shape[2] == 3:  # Если jpg, то преобразуем в float
    new_img = np.array(new_img) / 255
    imsave('butterfly_N36_alpha1.jpg', new_img)
