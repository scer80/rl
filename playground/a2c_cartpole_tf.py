import gym
import tensorflow as tf

import argparse
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

class PolicyNet(): # a.k.a. actor
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
			self.policy = tf.nn.softmax(self.logits)#tf.clip_by_value(self.logits, -1e10, +1e10))

			self.loss = tf.reduce_mean(tf.multiply(-tf.log(self.policy+1e-10), self.advantage))
			# self.loss = tf.reduce_mean(
			# 		tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.advantage)
			# 	)
			self.optimizer = tf.train.AdamOptimizer(self.lr)
			self.train_step = self.optimizer.minimize(self.loss)

class ValueNet(): # a.k.a. critic
	def __init__(self, state_shape=(4), fc_layer_sizes=[100, 100], name_scope='value_net'):
		self.name_scope = name_scope
		with tf.variable_scope(self.name_scope):
			self.state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape))
			self.target = tf.placeholder(dtype=tf.float32, shape=(None, ))
			self.lr = tf.placeholder(dtype=tf.float32)
	
			self.layer = tf.contrib.layers.flatten(self.state)
			for layer_size in fc_layer_sizes:
				self.layer = tf.contrib.layers.fully_connected(self.layer, layer_size, activation_fn=tf.nn.relu)
			self.value = tf.contrib.layers.fully_connected(self.layer, 1, activation_fn=None)

			self.loss = tf.reduce_mean(tf.squared_difference(self.value, self.target))			
			self.optimizer = tf.train.AdamOptimizer(self.lr)
			self.train_step = self.optimizer.minimize(self.loss)			

def a2c_alg(args):
	env = gym.make('CartPole-v1')	

	value_losses = np.zeros((args.episodes))
	policy_losses = np.zeros((args.episodes))
	total_rewards = np.zeros((args.episodes))

	actor = PolicyNet(state_shape=env.observation_space.shape, action_size=env.action_space.n, fc_layer_sizes=[24])
	critic = ValueNet(state_shape=env.observation_space.shape, fc_layer_sizes=[24])

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	# learn the policy probability distribution 
	with tqdm(desc='Episode', total=args.episodes, unit=' episodes') as pbar:
		for episode in range(args.episodes):
			# reset the environment
			state = env.reset()
			loss_sum = 0.0
			for step_idx in range(args.max_episode_length):
				# determine policy
				policy = actor.policy.eval(feed_dict={actor.state: state[np.newaxis,:]})[0]
				# select action according to policy
				action = np.random.choice(env.action_space.n, p=policy)
				assert all(policy>=0), "policy not >0 {}".format(policy)
				assert abs(sum(policy)-1)<0.001, "sum policy not 1 {}".format(sum(policy))
				# execute action, get reward and next state
				next_state, reward, done, info = env.step(action)
				# value for state
				value = critic.value.eval(feed_dict={critic.state: state[np.newaxis,:]})[0]
				# set targte and advantage
				advantage = np.zeros((1, env.action_space.n), dtype=np.float32)
				target = np.zeros((1, ), dtype=np.float32)				
				if done:
					reward = (step_idx - 498.0)/10.0 # 1.0 if (step_idx == 499) else -1.0 # 
					# target 
					target[0] = reward
					# advantage
					advantage[0, action] = reward - value
				else:
					# value for next state
					next_value = critic.value.eval(feed_dict={critic.state: next_state[np.newaxis,:]})[0]
					# target 
					target[0] = reward + args.gamma*next_value
					# advantage
					advantage[0, action] = reward + args.gamma*next_value - value
					# update total reward for episode
					total_rewards[episode] += reward
				# train policy network
				run_result = sess.run([actor.loss, actor.train_step, actor.optimizer._lr_t], 
					feed_dict={actor.state:state[np.newaxis,:], actor.advantage:advantage, actor.lr:args.actor_lr})
				policy_losses[episode] = run_result[0]
				args.actor_lr = run_result[2]
				# train validation network
				run_result = sess.run([critic.loss, critic.train_step, critic.optimizer._lr_t], 
					feed_dict={critic.state:state[np.newaxis,:], critic.target:target, critic.lr:args.critic_lr})
				value_losses[episode] = run_result[0]
				args.critic_lr = run_result[2]
				# update output
				pbar.set_description('Episode {:>5}'.format(episode))
				pbar.set_postfix(PLoss='{:>9.2f}'.format(policy_losses[episode]), VLoss='{:>9.2f}'.format(value_losses[episode]), 
					Reward='{:<5.1f}'.format(total_rewards[episode]))
					
				# update state
				state = next_state
				# end episode if done				
				if done:
					break

			pbar.update(1)

	display_plots(policy_losses, value_losses, total_rewards)

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
	parser = argparse.ArgumentParser(description='A2C Cartpole')
	parser.add_argument('--episodes', type=int, default=1000, metavar='E',
						help='number of training episodes (default: 1000)')
	parser.add_argument('--max-episode-length', type=int, default=500, metavar='L',
						help='maximum episode length (default: 500)')	
	# parser.add_argument('--sample-size', type=int, default=32, metavar='SS',
	# 					help='sample size for training (default: 32)')	
	parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
						help='discount factor (default: 0.99)')
	parser.add_argument('--critic-lr', type=float, default=0.005, metavar='vlr',
						help='critic (value) function learning rate (default: 0.005)')	
	parser.add_argument('--actor-lr', type=float, default=0.001, metavar='plr',
						help='actor (policy) learning rate (default: 0.001)')

	args = parser.parse_args()
	return args	

def main():
	args = arguments()
	a2c_alg(args)

if __name__ == '__main__':
	main()
