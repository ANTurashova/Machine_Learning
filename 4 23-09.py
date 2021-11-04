import matplotlib.pyplot as plt
import numpy as np
import pygame


(width, height) = (640, 480)  # размеры окна
bg_color = (255, 255, 255)  # цвет фона
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("#3")  # заголовок


points = np.empty((0, 2), dtype='f')

c_radius = 2
c_color = (0, 0, 255)
c_thickness = 0


# Цикл отображения окна
running = True
pushing = False
while running:
    # Обработка пользователльских событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pushing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            pushing = False

    if pushing:
        (x, y) = pygame.mouse.get_pos()
        r = np.random.uniform(0, 30)
        t = np.random.uniform(0, 2 * np.pi)
        coord = [x + r * np.cos(t), y + r * np.sin(t)]
        points = np.append(points, [coord], axis=0)

    screen.fill(bg_color)
    for point in points:
        pygame.draw.circle(screen, c_color, (int(point[0]), int(point[1])), c_radius, c_thickness)

    pygame.display.flip()

pygame.quit()

fig = plt.figure(figsize=(width/60, height/60))
plt.scatter(points[:, 0], points[:, 1], c="blue")
plt.show()
print(points.shape)
print(points)
