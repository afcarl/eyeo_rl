import numpy as np
from gym.spaces import Discrete

N_BINS = 10

def make_bins(low, high, n_bins=N_BINS):
    range = high - low
    bin_step = range/n_bins
    return np.arange(low, high + bin_step, bin_step)


class QLearner():
    def __init__(self, env, discount=0.9, learning_rate=0.8):
        """
        requires discrete action spaces.
        - discount: how much the agent values future rewards over immediate rewards
        - learning_rate: how quickly the agent learns. For deterministic environments (like ours), this should be left at 1
        """
        # initialize
        action_space = env.action_space
        observation_space = env.observation_space
        if not isinstance(action_space, Discrete):
            raise Exception('Action space must be Discrete')

        if not isinstance(observation_space, Discrete):
            lows = observation_space.low
            highs = observation_space.high
            self.bins = [make_bins(l, h) for l, h in zip(lows, highs)]
            n_states = N_BINS ** observation_space.shape[0]
        else:
            n_states = observation_space.n

        self.env = env
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = np.zeros((n_states, action_space.n))

    def _encode_observation(self, obs):
        if hasattr(self, 'bins'):
            binned = [np.digitize(v, bins) for bins, v in zip(self.bins, obs)]
            return int(''.join([str(b) for b in binned]))
        else:
            return obs

    def decide(self, obs, episode, total_episodes):
        obs = self._encode_observation(obs)
        decay = min(episode/(total_episodes/2), 1) # tweak
        rand = np.random.randn(1, self.env.action_space.n)
        return np.argmax(self.Q[obs] + (rand * (1 - decay)))

    def learn(self, prev_obs, next_obs, action, reward):
        prev_obs = self._encode_observation(prev_obs)
        next_obs = self._encode_observation(next_obs)
        self.Q[prev_obs, action] += self.learning_rate * ((reward + self.discount * self.Q[next_obs].max()) - self.Q[prev_obs, action])
