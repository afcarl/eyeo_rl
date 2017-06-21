import gym
from tqdm import tqdm
from table import QLearner
from deep import DQNLearner
import matplotlib.pyplot as plt

RENDER = False
EPISODES = 1000
MAXSTEPS = 100
LEARNER = 'table' # or 'deep'
ENV = 'FrozenLake-v0' # 'CartPole-v0'

# show available environments
# also see: <https://gym.openai.com/envs>
# print(gym.envs.registry.all())

if __name__ == '__main__':
    env = gym.make(ENV)

    if LEARNER == 'deep':
        agent = DQNLearner(env)
    else:
        agent = QLearner(env, discount=0.9, learning_rate=0.9)
    acc_reward = 0
    rewards = []
    for i in tqdm(range(EPISODES)):
        obs = env.reset()
        for t in range(MAXSTEPS):
            if RENDER:
                env.render()
            action = agent.decide(obs, i, EPISODES)
            new_obs, reward, done, info = env.step(action)
            agent.learn(obs, new_obs, action, reward)
            obs = new_obs
            acc_reward += reward
            if done:
                rewards.append(acc_reward/(i+1))
                break
    plt.plot(rewards)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
