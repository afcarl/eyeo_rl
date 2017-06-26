# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pygame
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 20)


class CatcherGame():
    def __init__(self, cell_size=40, size=(10, 10)):
        self.cell_size = cell_size
        self.width, self.height = size
        self.size = self.width * self.cell_size, self.height * self.cell_size
        self.action_space = [-1, 1, 0] # L/R/S

        self.paddle_padding = 1

        self.reset()
        self.observation_space = np.shape(self.observe())

    def reset(self):
        self.target = (np.random.randint(self.width - 1), 0)
        self.paddle = round(self.width/2)
        return self.observe()

    def observe(self):
        map = np.zeros((self.width, self.height), dtype=int)
        map[self.target] = 1
        map[self.paddle, self.height - 1] = 1
        for i in range(1 + self.paddle_padding*2):
            pos = self.paddle - self.paddle_padding + i
            map[pos, self.height - 1] = 1
        return map.reshape(self.width, self.height, 1)

    def step(self, action):
        action = self.action_space[action]
        self.move(action)
        tx, ty = self.target
        ty += 1
        if ty >= self.height - 1:
            done = True
            if abs(tx -  self.paddle) <= self.paddle_padding:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
            done = False
        self.target = tx, ty
        next_state = self.observe()
        return next_state, reward, done

    def move(self, action):
        self.paddle += action
        self.paddle = max(self.paddle_padding, self.paddle)
        self.paddle = min(self.width - 1 - self.paddle_padding, self.paddle)

    def render(self, info, policy=None):
        if not hasattr(self, 'screen'):
            self.screen = pygame.display.set_mode(self.size)

        # TODO

        pygame.display.update()
