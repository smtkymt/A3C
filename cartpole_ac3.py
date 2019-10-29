import argparse

from a3c import MasterAgent

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game Cartpole.')
parser.add_argument('--algorithm', default='a3c', type=str, help='Choose between \'a3c\' and \'random\'.')
parser.add_argument('--train', dest='train', action='store_true', help='Train our model.')
parser.add_argument('--lr', default=0.001, help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int, help='How often to update the global model.')
parser.add_argument('--max-eps', default=1000, type=int, help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99, help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='/tmp/', type=str, help='Directory in which to save the model.') 

args = parser.parse_args()

master = MasterAgent(args.save_dir, args.lr)
    
if args.train:
    master.train(args.algorithm, args.max_eps, args.update_freq, args.gamma)
else:
    master.play()
