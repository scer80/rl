import gym
import tensorflow as tf

import time
import threading
import argparse
import numpy as np
from tqdm import tqdm

import pylab
import matplotlib.pyplot as plt

episode = 0
scores = []

class A3CNet():
	def __init__(self, state_shape=[4], action_size=2, common_layers=[32], actor_layers=[32], critic_layers=[32]):
		self.state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape))
		self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, action_size))
		self.value_target = tf.placeholder(dtype=tf.float32, shape=(None, ))
		self.actor_lr = tf.placeholder(dtype=tf.float32)
		self.critic_lr = tf.placeholder(dtype=tf.float32)

		self.layer = tf.contrib.layers.flatten(self.state)
		for layer_size in common_layers:
			self.layer = tf.contrib.layers.fully_connected(self.layer, layer_size, activation_fn=tf.nn.relu)

		self.actor_layer = self.layer
		for layer_size in actor_layers:
			self.actor_layer = tf.contrib.layers.fully_connected(self.actor_layer, layer_size, activation_fn=tf.nn.relu)

		self.logits = tf.contrib.layers.fully_connected(self.actor_layer, action_size, activation_fn=None)
		self.policy = tf.nn.softmax(self.logits) #tf.clip_by_value(self.logits, -1e8, +1e8))

		self.actor_loss = tf.reduce_mean(tf.multiply(-tf.log(self.policy+1e-10), self.advantage))
		self.actor_train_step = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss)		

		self.critic_layer = self.layer
		for layer_size in critic_layers:
			self.critic_layer = tf.contrib.layers.fully_connected(self.critic_layer, layer_size, activation_fn=tf.nn.relu)

		self.value = tf.contrib.layers.fully_connected(self.critic_layer, 1, activation_fn=None)

		self.critic_loss = tf.reduce_mean(tf.squared_difference(self.value, self.value_target))
		self.critic_train_step = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)

class Agent(threading.Thread):
	def __init__(self, a3cnet, sess, args):
		threading.Thread.__init__(self)
		self.a3cnet = a3cnet
		self.sess = sess
		self.args = args

	def run(self):
		args = self.args
		sess = self.sess
		env = gym.make('CartPole-v1')	
	
		states = np.zeros((args.max_episode_length, *env.observation_space.shape), dtype=np.float32)
		actions = np.zeros((args.max_episode_length, ), dtype=np.int32)
		rewards = np.zeros((args.max_episode_length, ), dtype=np.float32)
		discounted_utilities = np.zeros((args.max_episode_length, ), dtype=np.float32)

		with sess.as_default():	
		# with tqdm(desc='Episode', total=args.episodes, unit=' episodes') as pbar:
			assert tf.get_default_session() is sess, 'session mismatch'
			# for episode in range(args.episodes):
			global episode			
			while episode < args.episodes:
				# reset total_rewards
				total_rewards = 0
				# reset the environment
				state = env.reset()
				loss_sum = 0.0
				for step_idx in range(args.max_episode_length):
					# determine policy
					policy = self.a3cnet.policy.eval(feed_dict={self.a3cnet.state: state[np.newaxis,:]})[0]
					assert all(policy>=0), "policy not >0 {}".format(policy)
					assert abs(sum(policy)-1)<1e4, "sum policy not 1 {}".format(sum(policy))
					# select action according to policy
					action = np.random.choice(env.action_space.n, p=policy)
					# execute action, get reward and next state
					next_state, reward, done, info = env.step(action)
					# update sum_of_rewards
					total_rewards += reward
					# save state, action, reward
					states[step_idx,:] = state
					actions[step_idx] = action
					rewards[step_idx] = reward
					# update state
					state = next_state
					# end episode if done				
					if done:
						break

				# save sum_of_rewards
				scores.append(total_rewards) 

				# learning
				eps_len = step_idx+1
				if eps_len < 500:
					running_reward = -100.0 if done else self.a3cnet.value.eval(feed_dict={self.a3cnet.state: next_state[np.newaxis,:]})[0]
					for step_back in reversed(range(eps_len)):
						running_reward = args.gamma*running_reward + rewards[step_back]
						advantage = np.zeros((1, env.action_space.n))
						state = states[np.newaxis,step_back,:]
						value = self.a3cnet.value.eval(feed_dict={self.a3cnet.state: state})[0]
						advantage[0,actions[step_back]] = running_reward - value					
						# train policy network
						run_result = sess.run([self.a3cnet.actor_loss, self.a3cnet.actor_train_step], 
							feed_dict={self.a3cnet.state:state, self.a3cnet.advantage:advantage, self.a3cnet.actor_lr:args.actor_lr})
						# train validation network
						run_result = sess.run([self.a3cnet.critic_loss, self.a3cnet.critic_train_step], 
							feed_dict={self.a3cnet.state:state, self.a3cnet.value_target:[running_reward], self.a3cnet.critic_lr:args.critic_lr})

				episode += 1

def a3c_alg(args):
	a3cnet = A3CNet()

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	agents = [Agent(a3cnet, sess, args) for i in range(args.number_of_agents)]

	for agent in agents:
		agent.start()

	while True:
		time.sleep(1)

		copy_of_scores = scores[:]
		plt.plot(copy_of_scores, 'b')
		plt.savefig("./save_graph/cartpole_a3c_mycode.png")		

		if all(not agent.isAlive() for agent in agents):
			break

	print('All done.')



def display_plots(policy_losses, value_losses, total_rewards):
	assert policy_losses.shape == total_rewards.shape, \
		'policy_losses.shape != total_rewards.shape ({} != {})'.format(policy_losses.shape, total_rewards.shape)
	assert value_losses.shape == total_rewards.shape, \
		'value_losses.shape != total_rewards.shape ({} != {})'.format(value_losses.shape, total_rewards.shape)

	n = total_rewards.shape[0]

	fig_unit_size = 5.0
	fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(1.5*fig_unit_size,1.*fig_unit_size))

	ax1.plot(policy_losses, color='red')
	ax1.plot(value_losses, color='orange')
	ax1.set_ylim([0, max(value_losses.max(), policy_losses.max())*1.05])
	ax1.set_ylabel('Loss', color='red')

	ax2 = ax1.twinx()

	ax2.plot(total_rewards, color='blue')
	ax2.set_ylim([0, total_rewards.max()*1.05])
	ax2.set_ylabel('Total rewards', color='blue')

	plt.show()	

def arguments():
	parser = argparse.ArgumentParser(description='A3C Cartpole')
	parser.add_argument('--episodes', type=int, default=2000, metavar='E',
						help='number of training episodes (default: 2000)')
	parser.add_argument('--max-episode-length', type=int, default=500, metavar='L',
						help='maximum episode length (default: 500)')	
	parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
						help='discount factor (default: 0.99)')
	parser.add_argument('--actor-lr', type=float, default=0.001, metavar='plr',
						help='actor (policy) learning rate (default: 0.001)')
	parser.add_argument('--critic-lr', type=float, default=0.005, metavar='vlr',
						help='critic (value) function learning rate (default: 0.005)')	
	parser.add_argument('--number-of-agents', type=int, default=4, metavar='na',
						help='number of agents (default: 4)')

	args = parser.parse_args()
	return args	

def main():
	args = arguments()
	a3c_alg(args)

if __name__ == '__main__':
	main()