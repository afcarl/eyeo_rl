import random
import numpy as np
from collections import deque
from gym.spaces import Discrete
from keras.models import Sequential
from keras.layers.core import Dense


class DQNLearner():
    def __init__(self, env, discount=1., explore=1., hidden_size=100, memory_limit=5000, batch_size=256):
        if not isinstance(env.action_space, Discrete):
            raise Exception('Action space must be Discrete')

        if isinstance(env.observation_space, Discrete):
            obs_dim = 1
        else:
            obs_dim = env.observation_space.shape[0]

        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(obs_dim,), activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(env.action_space.n))
        model.compile(loss='mse', optimizer='sgd')
        self.Q = model

        # experience replay:
        # remember states to "reflect" on later
        self.memory = deque([], maxlen=memory_limit)

        self.env = env
        self.explore = explore
        self.discount = discount
        self.batch_size = batch_size

    def _encode_observation(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array([obs])
        return obs.reshape((1, -1))

    def decide(self, obs, episode, total_episodes):
        obs = self._encode_observation(obs)
        decay = min(episode/(total_episodes/2), 1) # tweak
        if np.random.rand() <= (self.explore * (1 - decay)):
            return self.env.action_space.sample()
        q = self.Q.predict(obs)
        return np.argmax(q[0])

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
