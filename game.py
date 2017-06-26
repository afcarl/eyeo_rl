# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import pygame
import numpy as np
from itertools import product

pygame.init()
font = pygame.font.Font('arial.ttf', 20)


class Game():
    def __init__(self, objects, probs, cell_size=40, size=(2, 2), state_type='position'):
        self.cell_size = cell_size
        self.width, self.height = size
        self.size = self.width * self.cell_size, self.height * self.cell_size
        self.map = np.zeros((self.width, self.height), dtype=int)
        self.action_space = [(-1, 0), (1, 0), (0, -1), (1, 0)] # L/R/U/D

        self.objects = [{
            'name': 'empty',
            'color': (175, 175, 175),
            'reward': 0
        }]
        self.objects.extend(objects)
        probs.insert(0, 1.0 - sum(probs))
        assert sum(probs) == 1.0
        self.probs = probs

        self.rewards = [o['reward'] for o in self.objects]

        # initialize map
        self.generate_map()

        self.state_type = state_type
        if self.state_type == 'position':
            self.n_states = self.width * self.height
        elif self.state_type == 'world':
            self.n_states = len(self.objects)**(self.width * self.height)
        else:
            raise Exception('Unrecognized state type')

        self.reset()
        self.observation_space = np.shape(self.observe())

    def generate_map(self):
        for x, y in product(range(self.width), range(self.height)):
            obj = np.random.choice(self.objects, p=self.probs)
            idx = self.objects.index(obj)
            self.map[x, y] = idx
        self._map = np.copy(self.map)

    def reset(self):
        self.map = np.copy(self._map)
        self.agent_pos = (
            np.random.randint(self.width),
            np.random.randint(self.height))
        return self.observe()

    def observe(self):
        if self.state_type == 'position':
            x, y = self.agent_pos
            return y * self.width + x
        elif self.state_type == 'world':
            state = np.copy(self.map)
            state[self.agent_pos] = -1
            return state.reshape(20, 20, 1)

    def step(self, action):
        action = self.action_space[action]
        idx = self.move(action)
        reward = self.rewards[idx] - 1 # -1 per tick
        done = self.objects[idx].get('terminal', False)
        next_state = self.observe()
        return next_state, reward, done

    def move(self, action):
        x, y = self.agent_pos
        x += action[0]
        y += action[1]
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        self.agent_pos = x, y
        idx = self.map[x, y]
        self.map[x, y] = 0 # set to empty
        return idx

    def render(self, info, policy=None):
        if not hasattr(self, 'screen'):
            self.screen = pygame.display.set_mode(self.size)
        self._render_map()
        self._render_agent(*self.agent_pos)

        text = ', '.join(['{}: {}'.format(k, v) for k, v in info.items()])
        label = font.render(text, True, (66, 134, 244))
        self.screen.blit(label, (0, 0))

        if policy is not None:
            self._render_policy(policy)

        pygame.display.update()

    def _render_map(self):
        for idx, val in np.ndenumerate(self.map):
            x, y = idx
            obj = self.objects[val]
            pygame.draw.rect(
                self.screen, obj['color'],
                (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))

    def _render_text(self, txt, pos, color=(66, 134, 244)):
        x, y = pos
        label = font.render(txt, True, color)
        rect = label.get_rect(center=(
            x*self.cell_size+self.cell_size/2,
            y*self.cell_size+self.cell_size/2))
        self.screen.blit(label, rect)

    def _render_agent(self, x, y):
        w = h = self.cell_size * 0.5
        pygame.draw.ellipse(
            self.screen, (0,0,255),
            (x*self.cell_size+w/2, y*self.cell_size+h/2, w, h))

    def _render_policy(self, policy):
        arrows = ['←', '→', '↑', '↓']
        for idx, val in np.ndenumerate(self.map):
            x, y = idx
            idx = y * self.width + x
            arr = arrows[np.argmax(policy[idx])]
            if np.max(policy[idx]) == 0:
                self._render_text(arr, (x, y), color=(200,200,200))
            else:
                self._render_text(arr, (x, y))