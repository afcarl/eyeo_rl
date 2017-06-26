import random
import numpy as np
from collections import deque
from keras.layers import Conv2D
from keras.optimizers import sgd
from keras.models import Sequential
from keras.layers.core import Dense, Flatten


class DQNLearner():
    def __init__(self, env, type='simple', discount=1., explore=1., hidden_size=100, memory_limit=2500, batch_size=256, learning_rate=0.2):
        model = Sequential()
        if type == 'conv':
            input_shape = (env.observation_space[0], env.observation_space[1], 1)
            model.add(Conv2D(32, (8, 8),
                            subsample=(4,4),
                            activation='relu',
                            border_mode='same',
                            input_shape=input_shape))
            model.add(Conv2D(64, (4, 4), subsample=(2,2), activation='relu', border_mode='same'))
            model.add(Conv2D(64, (3, 3), subsample=(1,1), activation='relu', border_mode='same'))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(len(env.action_space)))
            model.compile(loss='mse', optimizer='adadelta')
        else:
            model.add(Dense(hidden_size, input_shape=(env.height * env.width,), activation='relu'))
            model.add(Dense(hidden_size, activation='relu'))
            model.add(Dense(len(env.action_space)))
            model.compile(loss='mse', optimizer=sgd(lr=learning_rate))
        self.Q = model
        self.type = type

        # experience replay:
        # remember states to "reflect" on later
        self.memory = deque([], maxlen=memory_limit)

        self.env = env
        self.explore = explore
        self.discount = discount
        self.batch_size = batch_size

    def _encode_observation(self, obs):
        if self.type == 'conv':
            return np.array([obs])
        else:
            return obs.reshape((1, -1))

    def decide(self, obs, episode, total_episodes):
        obs = self._encode_observation(obs)
        decay = min(episode/(total_episodes/2), 1) # tweak
        pred = self.Q.predict(obs)
        rand = np.random.randn(1, len(self.env.action_space)) * self.explore
        return np.argmax(pred[0] + (rand * (1 - decay)))

    def learn(self, prev_obs, next_obs, action, reward):
        prev_obs = self._encode_observation(prev_obs)
        next_obs = self._encode_observation(next_obs)
        self.remember(prev_obs, next_obs, action, reward)
        loss = self.replay(self.batch_size)
        return loss

    def remember(self, state, next_state, action, reward):
        # the deque object will automatically keep a fixed length
        self.memory.append((state, next_state, action, reward))

    def _prep_batch(self, batch_size):
        if batch_size > self.memory.maxlen:
            Warning('batch size should not be larger than max memory size. Setting batch size to memory size')
            batch_size = self.memory.maxlen

        batch_size = min(batch_size, len(self.memory))

        inputs = []
        targets = []

        # prep the batch
        # inputs are states, outputs are values over actions
        batch = random.sample(list(self.memory), batch_size)
        random.shuffle(batch)
        for state, next_state, action, reward in batch:
            inputs.append(state)
            target = self.Q.predict(state)[0]

            # non-zero reward indicates terminal state
            if reward:
                target[action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                Q_sa = np.max(self.Q.predict(next_state)[0])
                target[action] = reward + self.discount * Q_sa
            targets.append(target)

        # to numpy matrices
        return np.vstack(inputs), np.vstack(targets)

    def replay(self, batch_size):
        inputs, targets = self._prep_batch(batch_size)
        loss = self.Q.train_on_batch(inputs, targets)
        return loss

    def save(self, fname):
        self.Q.save_weights(fname)

    def load(self, fname):
        self.Q.load_weights(fname)
