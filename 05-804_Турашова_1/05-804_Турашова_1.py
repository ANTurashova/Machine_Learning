# 1. Пиксельная маска части изображения
# Срок заканчивается 4 октября 2021 г., 23:59
# Инструкции
# Дано:
# - изображение (формат jpg/png)
# - диапазон строк для маски (y_min, y_max)
# - диапазон столбцов для маски (x_min, x_max)
# - число разбиений на области (h, например, для h=10 маска разбивается на 10*10=100 равных областей)
#
# Требуется:
# - написать алгоритм усреднения цвета в каждой области (усреднение по каждому цветовому каналу: R,G,B)
# - прислать файл(ы): исходное изображение, изображение с маской, код (для работы в формате ipynb - всё в одном файле,
# в противном случае - все файлы поместить в один архив)
# - назвать файл по шаблону на латинице без пробелов: Группа_Фамилия_НомерРаботы (например, 11-000_Ivanov_1)

import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave

orig_img = imread('kitty.jpg')
img = orig_img.copy()
y_min = 20
y_max = 470
x_min = 390
x_max = 700
h = 10

move_on_x = (x_max - x_min) / h
move_on_y = (y_max - y_min) / h


def get_color_of_area(array_in_area):
    sum_in_area = [0, 0, 0]
    for i in array_in_area:
        for ii in i:
            sum_in_area = sum_in_area + ii
    return sum_in_area / len(array_in_area) / len(array_in_area[0])


for dy in range(h):
    y1 = int(y_min + dy * move_on_y)
    y2 = int(y_min + dy * move_on_y + move_on_y)
    for dx in range(h):
        x1 = int(x_min + dx * move_on_x)
        x2 = int(x_min + dx * move_on_x + move_on_x)
        img[y1:y2, x1:x2] = get_color_of_area(img[y1:y2, x1:x2])


plt.imshow(img)
plt.show()

imsave('kitty_new.jpg', img)
