#!/usr/bin/env python3
'''
Tester program for A3C

Adapted from https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py

Copyright (C) Simon D. Levy 2019

MIT License
'''

from a3c import A3CAgent, run_random

import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Run A3C algorithm on a gym game.')
parser.add_argument('--game', default='CartPole-v0', type=str, help='Choose game.')
parser.add_argument('--algorithm', default='ac3', type=str, help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true', help='Train our model.')
parser.add_argument('--lr', default=0.001, help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int, help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int, help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99, help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str, help='Directory in which to save the model.') 

args = parser.parse_args()

if args.algorithm == 'random':

    run_random(args.game, args.max_eps)

else:

    agent = A3CAgent(args.game, args.save_dir, args.lr)
        
    if args.train:

        moving_average_rewards = agent.train(args.max_eps, args.update_freq, args.gamma)

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.title(args.game)
        plt.savefig(os.path.join(args.save_dir, '{} Moving Average.png'.format(args.game)))
        plt.show()

    else:

        agent.play()
