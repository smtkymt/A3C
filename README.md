This repository contains a Python module and test code for running the Asynchronous Adavantage Actor Critic (A3C) algorithm 
for Deep Reinforcement Learning.  After reading Raymond Yuan's excellent A3C
[tutorial](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296),
I adapted his [CartPole code](https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py) as follows:

1. Translated the code to TensorFlow 2.0.

2. Added support for games other than CartPole.

##Installation

On Linux, you should run the following commands using sudo.

First you'll want to make sure your Python package installer (pip) is up-to-date

```pip3 install --upgrade pip```

