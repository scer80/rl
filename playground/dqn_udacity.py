import gym
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from collections import deque, namedtuple

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, 
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size, 
                                                            activation_fn=None)
            
            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

class DQN():

	def __init__(self, state_shape=(4), action_size=2, fc_layer_sizes=[100, 100]):
		self.inputs_ = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape))
		self.actions_ = tf.placeholder(dtype=tf.int32, shape=(None)) # selects Q-value for computation of the loss function
		self.targetQs_ = tf.placeholder(dtype=tf.float32, shape=(None))

		self.layer = tf.contrib.layers.flatten(self.inputs_)
		for layer_size in fc_layer_sizes:
			self.layer = tf.contrib.layers.fully_connected(self.layer, layer_size, activation_fn=tf.nn.relu)
		self.output = tf.contrib.layers.fully_connected(self.layer, action_size, activation_fn=None)

		self.Q = tf.reduce_sum(tf.multiply(self.output, tf.one_hot(self.actions_, action_size)), axis=1)
		self.state_not_zero = tf.logical_not(
								tf.reduce_all(
									tf.equal(self.inputs_, tf.zeros_like(self.inputs_)), 
									axis=1)
								)
		self.Qmax = tf.multiply(
						tf.reduce_max(self.output, axis=1),
						tf.cast(self.state_not_zero, tf.float32)
					)
		self.action_selected = tf.argmax(self.output, axis=1, output_type=tf.int32)
		self.loss = tf.reduce_mean(tf.squared_difference(self.targetQs_, self.Q))

		self.opt = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]

ReplayMemoryWord = namedtuple('ReplayMemoryWord', 'state action reward next_state')

class ReplayMemory():

	def __init__(self, maxlen):
		self.mem = deque([], maxlen)

	def append(self, s, a, r, s_):
		self.mem.append(ReplayMemoryWord(s, a, r, s_))

	def sample(self, nsamples):
		indices = np.random.choice(len(self.mem), nsamples, replace=False)

		states, actions, rewards, next_states = [], [], [], []
		for i in indices:
			states.append(self.mem[i].state)
			actions.append(self.mem[i].action)
			rewards.append(self.mem[i].reward)
			next_states.append(self.mem[i].next_state)									

		return states, actions, rewards, next_states        

train_episodes = 1000          # max number of episodes to learn from
max_steps = 200                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
explore_ep_end = 700
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 64               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 20                # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory


tf.reset_default_graph()
# mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)

env = gym.make('CartPole-v0')
# Initialize the simulation
env.reset()
# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())

mainQN = DQN(state_shape=env.observation_space.shape, action_size=env.action_space.n, fc_layer_sizes=[100, 100])
#memory = Memory(max_size=memory_size)
memory = ReplayMemory(memory_size)

# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulation
    # env.render()

    # Make a random action
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        #memory.add((state, action, reward, next_state))
        memory.append(state, action, reward, next_state)
        
        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        # Add experience to memory
        memory.append(state, action, reward, next_state)
        state = next_state

env.close()


# Now train with experiences
saver = tf.train.Saver()
rewards_list = []
loss = 0
explore_p = explore_start
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    step = 0
    with tqdm(desc='Episode', total=train_episodes, unit=' eps') as pbar:
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            state = env.reset()
            while t < max_steps:
                step += 1
                # Uncomment this next line to watch the training
                # env.render() 
                
                # Explore or Exploit
                # explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step) 
                # if explore_p > np.random.rand():
                #     # Make a random action
                #     action = env.action_space.sample()
                # else:
                #     # Get action from Q-network
                #     feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                #     Qs = sess.run(mainQN.output, feed_dict=feed)
                #     action = np.argmax(Qs)
                # select action
                if np.random.uniform() < explore_p:
                    action = np.random.choice(env.action_space.n)
                else:
                    action = mainQN.action_selected.eval(feed_dict={mainQN.inputs_: state[np.newaxis,:]})[0]                
                
                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)
        
                total_reward += reward

                pbar.set_description('Ep. {:5} step {:5}'.format(ep, t))
                pbar.set_postfix(Loss='{:4.2f}'.format(loss), Reward='{:3.1f}'.format(total_reward), 
                	p_explore='{:.2f}'.format(explore_p))
                pbar.update(1 if done else 0)
            
                if done:
                    # the episode ends so no next state
                    print('state = {}, next_state = {}'.format(state, next_state))
                    next_state = np.zeros(state.shape)
                    t = max_steps
                    
                    # print('Episode: {}'.format(ep),
                    #       'Total reward: {}'.format(total_reward),
                    #       'Training loss: {:.4f}'.format(loss),
                    #       'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))
                    
                    # Add experience to memory
                    memory.append(state, action, reward, next_state)
                    
                    # Start new episode
                    #env.reset()
                    # Take one random step to get the pole and cart moving
                    #state, reward, done, _ = env.step(env.action_space.sample())
    
                else:
                    # Add experience to memory
                    memory.append(state, action, reward, next_state)
                    state = next_state
                    t += 1
                
                # Sample mini-batch from memory
                # batch = memory.sample(batch_size)
                states, actions, rewards, next_states = memory.sample(batch_size)
                # states = np.array([each[0] for each in batch])
                # actions = np.array([each[1] for each in batch])
                # rewards = np.array([each[2] for each in batch])
                # next_states = np.array([each[3] for each in batch])
                
                # Train network
                # target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})
                target_Qs = sess.run(mainQN.Qmax, feed_dict={mainQN.inputs_: next_states})
                
                # Set target_Qs to 0 for states where episode ends
                #episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                #target_Qs[episode_ends] = (0, 0)
                #target_Qs[episode_ends] = (0)

                # targets = rewards + gamma * np.max(target_Qs, axis=1)
                targets = rewards + gamma * target_Qs
    
                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict={mainQN.inputs_: states,
                                               mainQN.targetQs_: targets,
                                               mainQN.actions_: actions})

            if ep < explore_ep_end:
                explore_p -= (explore_start - explore_stop)/explore_ep_end
        
    saver.save(sess, "checkpoints/cartpole.ckpt")


import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 


eps, rews = np.array(rewards_list).T
smoothed_rews = running_mean(rews, 10)
plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Reward')

plt.show()

