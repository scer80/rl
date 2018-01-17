import gym
import tensorflow as tf

import argparse
import numpy as np
from tqdm import tqdm
from collections import deque, namedtuple

import matplotlib.pyplot as plt

class ReinforcePolicyNet():

	def __init__(self, state_shape=(4), action_size=2, fc_layer_sizes=[100, 100], name_scope='policy_net'):
		self.name_scope = name_scope
		with tf.variable_scope(self.name_scope):
			self.state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape))
			self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, action_size))
			self.lr = tf.placeholder(dtype=tf.float32)
	
			self.layer = tf.contrib.layers.flatten(self.state)
			for layer_size in fc_layer_sizes:
				self.layer = tf.contrib.layers.fully_connected(self.layer, layer_size, activation_fn=tf.nn.relu)
			self.logits = tf.contrib.layers.fully_connected(self.layer, action_size, activation_fn=None)
			self.policy = tf.nn.softmax(self.logits)

			self.loss = tf.reduce_mean(tf.multiply(-tf.log(self.policy), self.advantage))
			self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

def reinforce_alg(args):
	env = gym.make('CartPole-v1')	

	losses = np.zeros((args.episodes))
	episode_lengths = np.zeros((args.episodes))
	total_rewards = np.zeros((args.episodes))

	pg_agent = ReinforcePolicyNet(state_shape=env.observation_space.shape, action_size=env.action_space.n, fc_layer_sizes=[24, 24])

	state = env.reset()
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	# learn the policy probability distribution 
	states = np.zeros((args.max_episode_length, *env.observation_space.shape), dtype=np.float32)
	actions = np.zeros((args.max_episode_length, ), dtype=np.int32)
	rewards = np.zeros((args.max_episode_length, ), dtype=np.float32)
	discounted_utilities = np.zeros((args.max_episode_length, ), dtype=np.float32)	
	with tqdm(desc='Episode', total=args.episodes, unit=' episodes') as pbar:
		for episode in range(args.episodes):
			loss_sum = 0.0
			for step_idx in range(args.max_episode_length):
				# determine policy
				policy = pg_agent.policy.eval(feed_dict={pg_agent.state: state[np.newaxis,:]})[0]
				# select action according to policy
				action = np.random.choice(env.action_space.n, p=policy)
				# execute action, get reward and next state
				next_state, reward, done, info = env.step(action)
				# if done, set big negative rewards
				if done:
					reward = -10.
				else:
					# update total reward for episode
					total_rewards[episode] += reward
				# save state, action, reward
				states[step_idx,:] = state
				actions[step_idx] = action
				rewards[step_idx] = reward
				# update state
				state = next_state
				# end episode if done				
				if done:
					break

			# compute discounted utilities
			eps_len = step_idx+1
			running_sum = 0.0
			for idx, reward in enumerate(reversed(rewards[:eps_len])):
				running_sum = args.gamma * running_sum + reward
				discounted_utilities[eps_len-1-idx] = running_sum

			discounted_utilities -= np.mean(discounted_utilities)
			discounted_utilities /= np.std(discounted_utilities)

			advantages = np.zeros((eps_len, env.action_space.n))
			for idx in range(eps_len):
				advantages[idx, actions[idx]] = discounted_utilities[idx]
			run_result = sess.run([pg_agent.loss, pg_agent.train_step], 
				feed_dict={pg_agent.state:states[:eps_len,:], pg_agent.advantage:advantages, pg_agent.lr:args.learning_rate})
			loss = run_result[0]
			losses[episode] = loss

			# update output
			pbar.set_description('Episode {:>5}'.format(episode))
			pbar.set_postfix(Loss='{:>9.2f}'.format(loss), Reward='{:<5.1f}'.format(total_rewards[episode]))
			pbar.update(1)

			# reset environment
			state = env.reset()

	display_plots(losses, total_rewards)

def display_plots(losses, total_rewards):
	assert losses.shape == total_rewards.shape, 'losses.shape != total_rewards.shape ({} != {})'.format(losses.shape, total_rewards.shape)

	n = losses.shape[0]

	fig_unit_size = 5.0
	fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(1.5*fig_unit_size,1.*fig_unit_size))

	ax1.plot(losses, color='red')
	ax1.set_ylim([0, losses.max()*1.05])
	ax1.set_ylabel('Loss', color='red')

	ax2 = ax1.twinx()

	ax2.plot(total_rewards, color='blue')
	ax2.set_ylim([0, total_rewards.max()*1.05])
	ax2.set_ylabel('Total rewards', color='blue')

	plt.show()	

def arguments():
	parser = argparse.ArgumentParser(description='Reinforce Cartpole')
	parser.add_argument('--episodes', type=int, default=1000, metavar='E',
						help='number of training episodes (default: 1000)')
	parser.add_argument('--max-episode-length', type=int, default=200, metavar='L',
						help='maximum episode length (default: 200)')	
	parser.add_argument('--sample-size', type=int, default=32, metavar='SS',
						help='sample size for training (default: 32)')	
	parser.add_argument('--gamma', type=float, default=0.9, metavar='g',
						help='discount factor (default: 0.9)')
	parser.add_argument('--learning-rate', type=float, default=0.01, metavar='lr',
						help='learning rate (default: 0.01)')	

	args = parser.parse_args()
	return args	

def main():
	args = arguments()
	reinforce_alg(args)

if __name__ == '__main__':
	main()
