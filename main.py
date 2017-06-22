import argparse
from game import Game
from tqdm import tqdm
from table import QLearner
from deep import DQNLearner
import matplotlib.pyplot as plt

"""
- draw policy
- label episode number
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('learner', choices=['deep', 'table'])
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
    game = Game([{
        'name': 'treasure',
        'color': (238, 244, 66),
        'reward': 10,
        'terminal': True
    }], probs=[0.1], size=(10, 10))

    if args.learner == 'deep':
        agent = DQNLearner(game,
                           discount=args.discount,
                           explore=args.explore,
                           hidden_size=args.hidden_size,
                           batch_size=args.batch_size,
                           memory_limit=args.memory_limit)
    else:
        agent = QLearner(game,
                         discount=args.discount,
                         learning_rate=args.learning_rate)

    acc_reward = 0
    rewards = []
    bar = tqdm(range(args.episodes))
    for i in bar:
        obs = game.reset()
        acc_loss = 0
        for t in range(args.max_steps):
            if args.render:
                game.render()
            action = agent.decide(obs, i, args.episodes)
            new_obs, reward, done = game.step(action)
            loss = agent.learn(obs, new_obs, action, reward)
            if loss:
                acc_loss += loss
            obs = new_obs
            acc_reward += reward
            if done:
                break
        avg_reward = acc_reward/(i+1)
        rewards.append(avg_reward)
        if acc_loss != 0:
            bar.set_postfix(loss=acc_loss, reward=avg_reward)
        else:
            bar.set_postfix(reward=avg_reward)
    plt.plot(rewards)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
