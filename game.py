import sys
import pygame
import numpy as np
from time import sleep
from itertools import product

pygame.init()
font = pygame.font.SysFont('arial', 20)


class Game():
    def __init__(self, objects, probs, cell_size=40, size=(2, 2), state_type='position'):
        self.cell_size = cell_size
        self.width, self.height = size
        self.size = self.width * self.cell_size, self.height * self.cell_size
        self.map = np.zeros((self.width, self.height), dtype=int)
        self.action_space = [(-1, 0), (1, 0), (0, -1), (1, 0)]

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
        for x, y in product(range(self.width), range(self.height)):
            obj = np.random.choice(self.objects, p=self.probs)
            idx = self.objects.index(obj)
            self.map[x, y] = idx
        self._map = np.copy(self.map)

        self.state_type = state_type
        if self.state_type == 'position':
            self.n_states = self.width * self.height
        elif self.state_type == 'world':
            self.n_states = len(list(product(range(len(self.objects)), repeat=(self.width * self.height))))
        else:
            raise Exception('Unrecognized state type')

        self.reset()

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
            state = self.map
            state[self.agent_pos] = -1
            return state.flatten()

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

    def render(self):
        if not hasattr(self, 'screen'):
            self.screen = pygame.display.set_mode(self.size)
        self._render_map()
        self._render_agent(*self.agent_pos)
        pygame.display.flip()

    def _render_map(self):
        for idx, val in np.ndenumerate(self.map):
            x, y = idx
            obj = self.objects[val]
            pygame.draw.rect(
                self.screen, obj['color'],
                (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))
            #draw_text(np.random.choice(['←', '↑', '→', '↓']), (x, y))

    def _render_text(self, txt, pos):
        x, y = pos
        label = font.render(txt, True, (66, 134, 244))
        rect = label.get_rect(center=(
            x*self.cell_size+self.cell_size/2,
            y*self.cell_size+self.cell_size/2))
        self.screen.blit(label, rect)

    def _render_agent(self, x, y):
        w = h = self.cell_size * 0.5
        pygame.draw.ellipse(
            self.screen, (0,0,255),
            (x*self.cell_size+w/2, y*self.cell_size+h/2, w, h))

    # def neighbors(self, x, y):
    #     """moore neighborhood values for coordinate"""
    #     xs, ys = [x], [y]
    #     if x > 0:
    #         xs.append(x-1)
    #     if x < self.width - 1:
    #         xs.append(x+1)
    #     if y > 0:
    #         ys.append(y-1)
    #     if y < self.height - 1:
    #         ys.append(y+1)
    #     return [self.map[x, y] for x, y in product(xs, ys)]


if __name__ == '__main__':
    objects = [{
        'name': 'fruit',
        'color': (255, 0, 0),
        'reward': 1
    }, {
        'name': 'pit',
        'color': (0, 0, 0),
        'reward': -100,
        'terminal': True
    }, {
        'name': 'treasure',
        'color': (238, 244, 66),
        'reward': 10,
        'terminal': True
    }]
    probs = [0.02, 0.02, 0.01]
    game = Game(objects, probs, state_type='world')
    print(game.n_states)

    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYUP and event.key == 113: sys.exit()
    #     game.reset()
    #     game.render()
    #     sleep(0.5)