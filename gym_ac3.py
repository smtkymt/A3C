#!/usr/bin/env python3
'''
Test script for A3C module.

Adapted from https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py

Copyright (C) Simon D. Levy 2019

This file is part of A3C.

A3C is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.
This code is distributed in the hope that it will be useful,     
but WITHOUT ANY WARRANTY without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU Lesser General Public License 
along with this code.  If not, see <http:#www.gnu.org/licenses/>.
'''

import os
import argparse
import gym
from a3c import A3CAgent, run_random
import matplotlib.pyplot as plt

if __name__ == '__main__':

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
