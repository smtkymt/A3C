#!/usr/bin/env python3
import os
import argparse
from queue import Queue
import gym
from a3c import A3CAgent, run_random
import matplotlib.pyplot as plt

GAME_NAME = 'CartPole-v0'

# A factory class for generating gym environments
class EnvironmentFactory:

    def __init__(self, name):

        self.name = name

    def createEnvironment(self):

        return gym.make(self.name)

    def getName(self):

        return self.name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run A3C algorithm on the game Cartpole.')
    parser.add_argument('--algorithm', default='a3c', type=str, help='Choose between \'a3c\' and \'random\'.')
    parser.add_argument('--train', dest='train', action='store_true', help='Train our model.')
    parser.add_argument('--lr', default=0.001, help='Learning rate for the shared optimizer.')
    parser.add_argument('--update-freq', default=20, type=int, help='How often to update the global model.')
    parser.add_argument('--max-eps', default=1000, type=int, help='Global maximum number of episodes to run.')
    parser.add_argument('--gamma', default=0.99, help='Discount factor of rewards.')
    parser.add_argument('--save-dir', default='/tmp/', type=str, help='Directory in which to save the model.') 

    args = parser.parse_args()

    factory = EnvironmentFactory(GAME_NAME)

    if args.algorithm == 'random':

        run_random(factory, args.max_eps)

    else:

        agent = A3CAgent(factory, args.save_dir, args.lr)
            
        if args.train:

            moving_average_rewards = agent.train(args.max_eps, args.update_freq, args.gamma)

            plt.plot(moving_average_rewards)
            plt.ylabel('Moving average ep reward')
            plt.xlabel('Step')
            plt.savefig(os.path.join(args.save_dir, '{} Moving Average.png'.format(GAME_NAME)))
            plt.show()

        else:

            agent.play()
