# 2. Генератор данных в pygame
# Срок заканчивается 4 октября 2021 г., 23:59
# Инструкции
# Дано:
# - код для генерации двумерного массива в pygame. Начало координат в окне pygame находится в левом верхнем углу
# - полученный массив выводится с помощью библиотеки matplotlib, где начало координат находится в левом нижнем углу
# - таким образом изображение в matplotlib выглядит отраженным по вертикали по отношению к исходному в pygame
#
# Требуется:
# - внести изменения в код генератора данных для корректировки координат точек массива так, чтобы множество точек в
# pygame и matplotlib выглядели одинаково (для этого нужно сделать преобразования координат в момент генерации точек
# и обратное преобразование - в момент отображение в pygame)
# - (не обязательно) убрать из массива точек повторяющиеся объекты (оставить только уникальные)
# - назвать файл по шаблону на латинице без пробелов: Группа_Фамилия_НомерРаботы (например, 11-000_Ivanov_2)

import matplotlib.pyplot as plt
import numpy as np
import pygame

(width, height) = (640, 480)
bg_color = (255, 255, 255)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("#3")

points = np.empty((0, 2), dtype='f')
c_radius = 2
c_color = (0, 0, 255)
c_thickness = 0


def adding_point(coord):
    global points

    # if coord not in points:  # Не корректно работает! "Проглатывает" многие координаты
    #     points = np.append(points, [coord], axis=0)

    presence = False
    for i in points:
        list_int_i = [int(item) for item in list(i)]
        if coord == list_int_i:
            presence = True
    if presence == False:
        points = np.append(points, [coord], axis=0)


def from_pygame_to_pyplot():
    global points
    for point in points:
        point[1] = height - point[1]


running = True
pushing = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pushing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            pushing = False

    if pushing:
        (x, y) = pygame.mouse.get_pos()
        coord = [x, y]
        adding_point(coord)

    screen.fill(bg_color)
    for point in points:
        pygame.draw.circle(screen, c_color, (int(point[0]), int(point[1])), c_radius, c_thickness)

    pygame.display.flip()
pygame.quit()

fig = plt.figure(figsize=(width/60, height/60))
from_pygame_to_pyplot()
plt.scatter(points[:, 0], points[:, 1], c="blue")
plt.show()
