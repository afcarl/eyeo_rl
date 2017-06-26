import pygame

pygame.init()


class BaseGame():
    font = pygame.font.Font('data/arial.ttf', 20)

    def reset(self):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError
