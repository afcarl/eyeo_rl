import numpy as np


class QLearner():
    def __init__(self, env, discount=0.9, learning_rate=0.8):
        """
        requires discrete action spaces.
        - discount: how much the agent values future rewards over immediate rewards
        - learning_rate: how quickly the agent learns. For deterministic environments (like ours), this should be left at 1
        """
        self.env = env
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = np.zeros((env.n_states, len(env.action_space)))

    def decide(self, obs, episode, total_episodes):
        decay = min(episode/(total_episodes/2), 1) # tweak
        rand = np.random.randn(1, len(self.env.action_space))
        return np.argmax(self.Q[obs] + (rand * (1 - decay)))

    def learn(self, prev_obs, next_obs, action, reward):
        self.Q[prev_obs, action] += self.learning_rate * ((reward + self.discount * self.Q[next_obs].max()) - self.Q[prev_obs, action])
