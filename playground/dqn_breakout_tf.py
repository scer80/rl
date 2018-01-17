import gym
import tensorflow as tf

import argparse
import numpy as np
from tqdm import tqdm
from collections import deque, namedtuple

import matplotlib.pyplot as plt

def breakout_random_control():
	env = gym.make("BreakoutDeterministic-v4")

	state_size = env.observation_space.shape
	action_size = env.action_space.n

	print('state_size = {}'.format(state_size))
	print('action_size = {}'.format(action_size))

	state = env.reset()

	sum_rewards = 0.0

	idx = 0
	while True:
		env.render()
		action = env.action_space.sample()
		action = (int(idx/6) % 2) + 2 if idx%6 > 0 else 1
		state, reward, done, info = env.step(action)

		sum_rewards += reward
		print('{:3}: action={}, R={}, sum_R={}, lives={}, done={}'.format(idx, action, reward, sum_rewards, info['ale.lives'], done), end='')

		if done:
			break

		input()
		idx += 1

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe		

def process_im(im):
	# crop
	im = im[32:192, :, :]
	# downsample
	im = im[::2, ::2, :]
	# normalize
	im = im/255.0
	# calculate luminance
	im = np.dot(im, [0.299, 0.587, 0.114])
	return im

def test_preprocessing():
	env = gym.make("BreakoutDeterministic-v4")
	state = env.reset()

	state_size = env.observation_space.shape
	action_size = env.action_space.n

	# run a few iterations
	for _ in range(30):
		action_sample = env.action_space.sample()
		state, reward, done, info = env.step(1)

	state = process_im(state)
	print('state_shape = {}'.format(state.shape))
	for k, v in info.items():
		print(k, v)
	plt.imshow(state, cmap='gray')
	plt.show()

class DQN():

	def __init__(self, state_shape=(80, 80, 4), action_size=3, name_scope='dqn'):
		self.name_scope = name_scope
		with tf.variable_scope(self.name_scope):
			self.state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape))
			self.action = tf.placeholder(dtype=tf.int32, shape=(None)) # selects Q-value for computation of the loss function
			self.targetQ = tf.placeholder(dtype=tf.float32, shape=(None))
	
			self.layer = tf.contrib.layers.conv2d(self.state, 32, 8, 4, padding='SAME', activation_fn=tf.nn.relu)
			self.layer = tf.contrib.layers.conv2d(self.layer, 64, 4, 2, padding='SAME', activation_fn=tf.nn.relu)
			self.layer = tf.contrib.layers.conv2d(self.layer, 64, 3, 1, padding='SAME', activation_fn=tf.nn.relu)
			self.layer = tf.contrib.layers.flatten(self.layer)
			self.layer = tf.contrib.layers.fully_connected(self.layer, 512, activation_fn=tf.nn.relu)
			self.Q = tf.contrib.layers.fully_connected(self.layer, action_size, activation_fn=None)
	
			self.Qa = tf.reduce_sum(tf.multiply(self.Q, tf.one_hot(self.action, action_size)), axis=1)
			self.Qmax = tf.reduce_max(self.Q, axis=1)
			self.action_selected = tf.argmax(self.Q, axis=1, output_type=tf.int32)
			self.loss = tf.reduce_mean(tf.squared_difference(self.targetQ, self.Qa))
	
			self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

	def get_weight_assign_ops(self, other_dqn):
		e1_params = [t for t in tf.trainable_variables() if t.name.startswith(self.name_scope)]
		e1_params = sorted(e1_params, key=lambda v: v.name)
		e2_params = [t for t in tf.trainable_variables() if t.name.startswith(other_dqn.name_scope)]
		e2_params = sorted(e2_params, key=lambda v: v.name)

		assigns = []
		for e1_v, e2_v in zip(e1_params, e2_params):
		    op = e1_v.assign(e2_v)
		    assigns.append(op)

		return assigns

class History():

	def __init__(self):
		self.shape = (80, 80, 4)
		self.content = np.zeros(self.shape, dtype=np.uint8)

	def process_state(self, state):
		# crop
		state = state[32:192, :, :]
		# downsample
		state = state[::2, ::2, :]
		# normalize
		state = state/255.0
		# calculate luminance
		state = np.dot(state, [0.299, 0.587, 0.114])
		return state

	def push(self, state):
		self.content[..., 1:] = self.content[..., :3]
		self.content[..., 0] = np.array(state*255.0, dtype=np.uint8)

	def get(self):
		return self.content.astype(np.float32)/255.0

	def get4(self):
		h = np.squeeze(self.content)
		h = np.rollaxis(h, 2)
		h = np.concatenate(h, 1)
		return h

def test_history():
	env = gym.make("BreakoutDeterministic-v4")
	state = env.reset()

	hist = History()

	# run a few iterations
	for idx in range(1000):
		hist.push(hist.process_state(state))
		action_sample = env.action_space.sample()
		state, reward, done, info = env.step((idx//10)%3+1)

		h = hist.get4()
		print(h.shape, type(h))

		fig = plt.figure(figsize=(16,4))
		plt.imshow(h, cmap='gray')
		plt.show()


ReplayMemoryWord = namedtuple('ReplayMemoryWord', 'state action reward next_state done')

class ReplayMemory():

	def __init__(self, maxlen):
		self.mem = deque([], maxlen)

	def append(self, s, a, r, s_, d):
		self.mem.append(ReplayMemoryWord(s, a, r, s_, d))

	def sample(self, nsamples):
		indices = np.random.choice(len(self.mem), nsamples, replace=False)

		states, actions, rewards, next_states, dones = [], [], [], [], []
		for i in indices:
			states.append(self.mem[i].state)
			actions.append(self.mem[i].action)
			rewards.append(self.mem[i].reward)
			next_states.append(self.mem[i].next_state)
			dones.append(self.mem[i].done)

		return states, actions, rewards, next_states, dones

def dqn_alg(args):
	env = gym.make('BreakoutDeterministic-v4')	

	losses = np.zeros((args.episodes))
	episode_lengths = np.zeros((args.episodes))
	total_rewards = np.zeros((args.episodes))

	state_shape = (80, 80, 4)
	action_size = 3
	dqn = DQN(state_shape=state_shape, action_size=action_size, name_scope='dqn')
	dqn_t = DQN(state_shape=state_shape, action_size=action_size, name_scope='dqn_t') # target DQN
	dqn_t_upd_ops = dqn_t.get_weight_assign_ops(dqn)
	mem = ReplayMemory(args.replay_memory_depth)
	hist = History()

	exploration_probability = 1.0
	state = env.reset()
	hist.push(hist.process_state(state))
	curr_h = hist.get()
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	# load replay memory with samples
	for idx in range(args.sample_size):
		action = np.random.choice(action_size) + 1
		next_state, reward, done, info = env.step(action)
		hist.push(hist.process_state(state))
		next_h = hist.get()

		mem.append(curr_h, action, reward, next_h, done)
		if done:
			next_state = env.reset()
		state = next_state
		curr_h = next_h
	# learn the Q function
	with tqdm(desc='Episode', total=args.episodes, unit=' episodes') as pbar:
		for episode in range(args.episodes):
			if episode % args.target_update_period == 0:
				sess.run(dqn_t_upd_ops)
			loss_sum = 0.0
			for step_idx in range(args.max_episode_length):
				# select action
				if np.random.uniform() < exploration_probability:
					action = np.random.choice(action_size) + 1
				else:
					action = dqn.action_selected.eval(feed_dict={dqn.state: curr_h[np.newaxis, ]})[0]
				# execute action, get reward and next state
				next_state, reward, done, info = env.step(action)
				hist.push(hist.process_state(state))
				next_h = hist.get()				
				# update total reward for episode
				total_rewards[episode] += reward
				# add s, a, r, s_, done to episodic memory
				mem.append(curr_h, action, reward, next_h, done)
				# sample episodic memory
				states, actions, rewards, next_states, dones = mem.sample(args.sample_size)
				# compute target Q values
				targetQmax = dqn_t.Qmax.eval(feed_dict={dqn_t.state: next_states})
				targetQ = np.array(rewards) + args.gamma * targetQmax
				# targetQ[np.array(dones)==False] += args.gamma * targetQmax[np.array(dones)==False]
				# train DQN
				run_result = sess.run([dqn.train_step, dqn.loss], feed_dict={dqn.state: states, dqn.action: actions, dqn.targetQ: targetQ})
				loss = run_result[1]
				loss_sum += loss
				# update state
				state = next_state
				curr_h = next_h
				# update output
				pbar.set_description('Episode {:>5} step {:>5}'.format(episode, step_idx))
				pbar.set_postfix(Loss='{:>9.2f}'.format(loss), Reward='{:<5.1f}'.format(total_rewards[episode]), 
					p_explore='{:>6.2f}'.format(exploration_probability))
				pbar.update(1 if done else 0)				
				# end episode if done
				if done:
					break
			
			episode_lengths[episode] = step_idx + 1
			losses[episode] = loss_sum / (step_idx + 1)
			# reset environment
			state = env.reset()
			# update exploration probability
			if episode < args.explore_decay_end:
				exploration_probability -= (1.0 - args.min_explore_prob)/args.explore_decay_end

	display_plots2(episode_lengths, losses, total_rewards)

def display_plots2(episode_lengths, losses, total_rewards):
	assert episode_lengths.shape == losses.shape, 'episode_lengths.shape != losses.shape ({} != {})'.format(episode_lengths.shape, losses.shape)
	assert episode_lengths.shape == total_rewards.shape, 'episode_lengths.shape != total_rewards.shape ({} != {})'.format(episode_lengths.shape, 
																													total_rewards.shape)

	n = episode_lengths.shape[0]

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
	parser = argparse.ArgumentParser(description='DQN Cartpole')
	parser.add_argument('--episodes', type=int, default=1000, metavar='E',
						help='number of training episodes (default: 1000)')
	parser.add_argument('--max-episode-length', type=int, default=10000, metavar='L',
						help='maximum episode length (default: 10000)')	
	parser.add_argument('--replay-memory-depth', type=int, default=1000, metavar='MD',
						help='replay memory depth (default: 1000)')
	parser.add_argument('--sample-size', type=int, default=32, metavar='SS',
						help='sample size for training (default: 32)')	
	parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
						help='discount factor (default: 0.99)')
	parser.add_argument('--min-explore-prob', type=float, default=0.01, metavar='ef',
						help='minimum exploration probability (default: 0.01)')
	parser.add_argument('--explore-decay-end', type=int, default=700, metavar='ed',
						help='number of episodes to reach minimal exploration probability (default: 700)')
	parser.add_argument('--target-update-period', type=int, default=2, metavar='tup',
						help='target update period (default: 2)')

	args = parser.parse_args()
	return args	

def main():
	args = arguments()
	# dqn_alg(args)
	# breakout_random_control()
	test_preprocessing()
	# test_history()

if __name__ == '__main__':
	main()
