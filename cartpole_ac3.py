import argparse
from queue import Queue
import gym
from a3c import A3CAgent, report

# Runs a random agent for baseline
def run_random(game_name, max_eps):
    env = gym.make('CartPole-v0')
    global_moving_average_reward = 0
    res_queue = Queue()
    reward_avg = 0
    for episode in range(max_eps):
        done = False
        env.reset()
        reward_sum = 0.0
        steps = 0
        while not done:
            # Sample randomly from the action space and step
            _, reward, done, _ = env.step(env.action_space.sample())
            steps += 1
            reward_sum += reward
        # Record statistics
        global_moving_average_reward = report(episode, max_eps, reward_sum, 0, global_moving_average_reward, res_queue, 0, steps) 
        reward_avg += reward_sum
    final_avg = reward_avg / float(max_eps)
    print("Average score across {} episodes: {}".format(max_eps, final_avg))

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

    if args.algorithm == 'random':

        run_random(args.max_eps)

    else:

        agent = A3CAgent('CartPole-v0', args.save_dir, args.lr)
            
        if args.train:
            agent.train(args.max_eps, args.update_freq, args.gamma)
        else:
            agent.play()
