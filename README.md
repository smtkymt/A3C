This repository contains a Python module and test code for running the Asynchronous Adavantage Actor Critic (A3C) algorithm 
for Deep Reinforcement Learning.  After reading Raymond Yuan's excellent A3C
[tutorial](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296),
I adapted his [CartPole code](https://github.com/tensorflow/models/blob/master/research/a3c_blogpost/a3c_cartpole.py) as follows:

1. Translated the code to TensorFlow 2.0.

2. Moved the test script into its own [module](https://github.com/simondlevy/a3c/blob/master/gym_ac3.py).

3. Added support for games other than CartPole.

