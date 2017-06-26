import sys
import pygame
import argparse
from tqdm import tqdm
from game import Game
from table import QLearner
from deep import DQNLearner
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('learner', choices=['deep', 'table'])
    parser.add_argument('-e', '--episodes', type=int, default=5000)
    parser.add_argument('-s', '--max-steps', type=int, default=50)
    parser.add_argument('-d', '--discount', type=float, default=1.)
    parser.add_argument('-x', '--explore', type=float, default=1.)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.9)
    parser.add_argument('-n', '--hidden-size', type=int, default=100)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-m', '--memory-limit', type=int, default=5000)
    parser.add_argument('-R', '--render', type=int, default=0)
    parser.add_argument('-D', '--debug', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    state_type = 'position' if args.learner == 'table' else 'world'

    game = Game([{
        'name': 'treasure',
        'color': (238, 244, 66),
        'reward': 100,
        'terminal': True
    }, {
        'name': 'fruit',
        'color': (255, 0, 0),
        'reward': 3
    }, {
        'name': 'pit',
        'color': (0, 0, 0),
        'reward': -100,
        'terminal': True
    }], probs=[0.05, 0.1, 0.05], size=(20, 20),
        state_type=state_type)
    # game.load()
    game.save()

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
        agent.load('model.h5')

    rewards = []
    acc_reward = 0
    avg_reward = 0
    bar = tqdm(range(args.episodes))
    for i in bar:
        acc_loss = 0
        ep_reward = 0
        obs = game.reset()
        for t in range(args.max_steps):
            if args.render:
                # if rendering, handle key events
                for event in pygame.event.get():
                    if event.type == pygame.KEYUP and event.key == 113:
                        pygame.quit()
                        sys.exit()

                # render an episode
                if i % args.render == 0:
                    policy = agent.Q if args.learner == 'table' else None
                    game.render({
                        'Episode': i,
                        'Reward': ep_reward,
                        'Avg Reward': '{:.1f}'.format(avg_reward)
                    }, policy=policy)

            # main training part
            action = agent.decide(obs, i, args.episodes)
            new_obs, reward, done = game.step(action)
            loss = agent.learn(obs, new_obs, action, reward)
            obs = new_obs

            ep_reward += reward
            if loss:
                acc_loss += loss
            if done:
                break

        # metrics
        acc_reward += ep_reward
        avg_reward = acc_reward/(i+1)
        rewards.append(avg_reward)
        if acc_loss != 0:
            bar.set_postfix(loss=acc_loss, reward=avg_reward)
        else:
            bar.set_postfix(reward=avg_reward)

    if args.learner == 'deep':
        agent.save('model.h5')

    plt.plot(rewards)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
