import gym
import tensorflow as tf

import argparse
import numpy as np
from tqdm import tqdm
from collections import deque, namedtuple

import matplotlib.pyplot as plt

class ReinforcePolicyNet():

	def __init__(self, state_shape=(80, 80, 1), action_size=6, name_scope='policy_net'):
		self.name_scope = name_scope
		with tf.variable_scope(self.name_scope):
			self.state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape))
			self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, action_size))
			self.lr = tf.placeholder(dtype=tf.float32)
	
			self.layer = tf.contrib.layers.conv2d(self.state, 32, 6, 3, padding='SAME', activation_fn=tf.nn.relu)
			self.layer = tf.contrib.layers.flatten(self.layer)
			self.layer = tf.contrib.layers.fully_connected(self.layer, 64, activation_fn=tf.nn.relu)
			self.layer = tf.contrib.layers.fully_connected(self.layer, 32, activation_fn=tf.nn.relu)
			self.logits = tf.contrib.layers.fully_connected(self.layer, action_size, activation_fn=None)
			self.policy = tf.nn.softmax(self.logits)

			#self.loss = tf.reduce_mean(tf.multiply(-tf.log(self.policy), self.advantage))
			self.loss = tf.reduce_mean(
							tf.nn.softmax_cross_entropy_with_logits(labels=self.advantage, logits=self.logits)
						)
			self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

def reinforce_alg(args):
	env = gym.make('Pong-v0')	

	losses = np.zeros((args.episodes))
	episode_lengths = np.zeros((args.episodes))
	total_rewards = np.zeros((args.episodes))

	state_shape = (80, 80, 1)
	pg_agent = ReinforcePolicyNet(state_shape=state_shape, action_size=6)

	state = env.reset()
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	# learn the policy probability distribution 
	states = np.zeros((args.max_episode_length, *state_shape), dtype=np.float32)
	probs = np.zeros((args.max_episode_length, env.action_space.n), dtype=np.int32)
	actions = np.zeros((args.max_episode_length, ), dtype=np.int32)
	rewards = np.zeros((args.max_episode_length, ), dtype=np.float32)
	discounted_utilities = np.zeros((args.max_episode_length, ), dtype=np.float32)	
	with tqdm(desc='Episode', total=args.episodes, unit=' episodes') as pbar:
		for episode in range(args.episodes):
			loss_sum = 0.0
			for step_idx in range(args.max_episode_length):
				# process state
				state = process_im(state)[..., np.newaxis]
				# determine policy
				policy = pg_agent.policy.eval(feed_dict={pg_agent.state: state[np.newaxis,:]})[0]
				# select action according to policy
				action = np.random.choice(env.action_space.n, p=policy)
				# execute action, get reward and next state
				next_state, reward, done, info = env.step(action)
				# update total reward for episode
				total_rewards[episode] += reward
				# save state, action, reward
				states[step_idx,:] = state
				probs[step_idx,:] = policy
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
				if reward != 0:
					running_sum = reward
				else:
					running_sum = args.gamma * running_sum + reward
				discounted_utilities[eps_len-1-idx] = running_sum
			discounted_utilities = 0.5 * (1 - discounted_utilities)

			Q = np.zeros((eps_len, env.action_space.n))
			V = np.zeros((eps_len, env.action_space.n))
			H = np.zeros((eps_len, env.action_space.n))

			for idx in range(eps_len):
				Q[idx, actions[idx]] = discounted_utilities[idx]
				V[idx, :] = probs[idx,:] * discounted_utilities[idx]
				H[idx, :] = probs[idx,:]

			labels = Q#args.entropy_weight * H + Q#(Q - V)
			run_result = sess.run([pg_agent.loss, pg_agent.train_step], 
				feed_dict={pg_agent.state:states[:eps_len,:], pg_agent.advantage:labels, pg_agent.lr:args.learning_rate})
			loss = run_result[0]
			losses[episode] = loss

			# update output
			# print(loss, total_rewards[episode])
			pbar.set_description('Episode {:>5}'.format(episode))
			pbar.set_postfix(Loss='{:>9.2f}'.format(loss), Reward='{:<5.1f}'.format(total_rewards[episode]))
			pbar.update(1)

			# reset environment
			state = env.reset()

	display_plots(losses, total_rewards)		

def process_im(im):
	# crop score, upper and lower walls
	im = im[34:194, :, :]
	# downsample
	im = im[::2, ::2, :]
	# normalize
	im = im/255.0
	# calculate luminance
	im = np.dot(im, [0.299, 0.587, 0.114])
	return im

def explore_pong():
	env = gym.make("Pong-v0")
	state = env.reset()

	state_size = env.observation_space.shape
	action_size = env.action_space.n

	print('state_size = {}'.format(state_size))
	print('action_size = {}'.format(action_size))

	print(dir(env.action_space))
	print(env.action_space.__doc__)

	print('min={}, max={}'.format(np.min(state), np.max(state)))

	# run a few iterations
	for _ in range(30):
		action_sample = env.action_space.sample()
		print('{} '.format(action_sample), end='')
		state, reward, done, info = env.step(0)
	print()

	state = process_im(state)
	print('state_shape = {}'.format(state.shape))
	plt.imshow(state, cmap='gray')
	plt.show()

def random_play_pong():
	env = gym.make("Pong-v0")
	state = env.reset()

	sum_rewards = 0.0

	while True:
		env.render()
		action = env.action_space.sample()
		state, reward, done, info = env.step(action)

		sum_rewards += reward
		print('R={}, sum_rewards={}, done={}'.format(reward, sum_rewards, done), end='')

		if done:
			break

		input()

def display_plots(losses, total_rewards):
	assert losses.shape == total_rewards.shape, 'losses.shape != total_rewards.shape ({} != {})'.format(losses.shape, total_rewards.shape)

	n = losses.shape[0]

	fig_unit_size = 5.0
	fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(1.5*fig_unit_size,1.*fig_unit_size))

	ax1.plot(losses, color='red')
	ax1.set_ylim([0, losses.max()*1.05])
	ax1.set_ylabel('Loss', color='red')

	ax2 = ax1.twinx()

	r_lim = max(abs(total_rewards.max()), abs(total_rewards.min()))
	ax2.plot(total_rewards, color='blue')
	ax2.set_ylim([-r_lim*1.05, r_lim*1.05])
	ax2.set_ylabel('Total rewards', color='blue')

	plt.show()	

def arguments():
	parser = argparse.ArgumentParser(description='Reinforce Pong')
	parser.add_argument('--episodes', type=int, default=1000, metavar='E',
						help='number of training episodes (default: 1000)')
	parser.add_argument('--max-episode-length', type=int, default=10000, metavar='L',
						help='maximum episode length (default: 10000)')	
	parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
						help='discount factor (default: 0.99)')
	parser.add_argument('--learning-rate', type=float, default=0.01, metavar='lr',
						help='learning rate (default: 0.01)')
	parser.add_argument('--entropy-weight', type=float, default=0.1, metavar='hw',
						help='entropy weight (default: 0.1)')	

	args = parser.parse_args()
	return args	

def main():
	args = arguments()
	reinforce_alg(args)
	# explore_pong()
	# random_play_pong()

if __name__ == '__main__':
	main()


