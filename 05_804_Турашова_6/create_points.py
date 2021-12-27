import pygame
import numpy as np


def main():
    width, height = (500, 500)
    screen = pygame.display.set_mode((width, height))

    points = np.empty((0, 2), dtype='f')

    running = True
    pressing = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pressing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                pressing = False

        if pressing and np.random.uniform(0, 1) > 0:
            (x, y) = pygame.mouse.get_pos()
            r = np.random.uniform(0, 30)
            phi = np.random.uniform(0, 2 * np.pi)
            coord = [x + r * np.cos(phi), height - y + r * np.sin(phi)]
            points = np.append(points, [coord], axis=0)

        screen.fill((255, 255, 255))
        for point in points:
            pygame.draw.circle(screen, (0, 0, 0), (int(point[0]), height - int(point[1])), 4, 0)

        pygame.display.flip()

    pygame.quit()
    return points
