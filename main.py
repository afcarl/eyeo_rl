import gym
import argparse
from tqdm import tqdm
from table import QLearner
from deep import DQNLearner
import matplotlib.pyplot as plt

# show available environments
# also see: <https://gym.openai.com/envs>
# print(gym.envs.registry.all())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('learner', choices=['deep', 'table'])
    parser.add_argument('env', choices=['FrozenLake-v0', 'CartPole-v0'])
    parser.add_argument('-e', '--episodes', type=int, default=5000)
    parser.add_argument('-s', '--max-steps', type=int, default=200)
    parser.add_argument('-d', '--discount', type=float, default=0.9)
    parser.add_argument('-x', '--explore', type=float, default=1.)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.8)
    parser.add_argument('-n', '--hidden-size', type=int, default=100)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-m', '--memory-limit', type=int, default=5000)
    parser.add_argument('-D', '--debug', action='store_true')
    parser.add_argument('-R', '--render', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    env = gym.make(args.env)
    if args.learner == 'deep':
        agent = DQNLearner(env,
                           discount=args.discount,
                           explore=args.explore,
                           hidden_size=args.hidden_size,
                           batch_size=args.batch_size,
                           memory_limit=args.memory_limit)
    else:
        agent = QLearner(env,
                         discount=args.discount,
                         learning_rate=args.learning_rate)

    acc_reward = 0
    rewards = []
    bar = tqdm(range(args.episodes))
    for i in bar:
        obs = env.reset()
        acc_loss = 0
        for t in range(args.max_steps):
            if args.render:
                env.render()
            action = agent.decide(obs, i, args.episodes)
            new_obs, reward, done, info = env.step(action)
            loss = agent.learn(obs, new_obs, action, reward)
            if loss:
                acc_loss += loss
            obs = new_obs
            acc_reward += reward
            if done:
                avg_reward = acc_reward/(i+1)
                rewards.append(avg_reward)
                break
        if acc_loss != 0:
            bar.set_postfix(loss=acc_loss, reward=avg_reward)
    plt.plot(rewards)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
