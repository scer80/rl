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

    def __init__(self, state_shape=(80, 80, 1), action_size=3, 
                seq_batch_size=32, seq_len=8, lstm_size=128, lstm_ncells=2,
                alpha=0.2, beta1=0.2, smooth=0.9,
                build=True):
        self.state_shape = state_shape
        self.action_size = action_size

        self.alpha = alpha
        self.beta1 = beta1
        self.smooth = smooth
        
        self.seq_len = seq_len
        self.lstm_size = lstm_size
        self.lstm_ncells = lstm_ncells
        
        # graph config
        self.gc = dict()
        
        if build:
            self.build()
            # Save names
            self.set_gc_names()

    def placeholders(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, *self.state_shape))
        self.actions = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.value_target = tf.placeholder(dtype=tf.float32, shape=(None, ))
        self.ph_lstm_state = tf.placeholder(dtype=tf.float32, shape=(self.lstm_ncells, 2, 1, self.lstm_size))
        self.lr = tf.placeholder(dtype=tf.float32)
        self.coeff_p = tf.placeholder(dtype=tf.float32)
        self.coeff_v = tf.placeholder(dtype=tf.float32)
        self.coeff_h = tf.placeholder(dtype=tf.float32)

    def zero_lstm_state(self):
        return np.zeros(dtype=np.float32, shape=(self.lstm_ncells, 2, 1, self.lstm_size))

    def build(self):
        self.placeholders()

        with tf.variable_scope('a3c'):
            self.layer = tf.contrib.layers.conv2d(self.state, 16, 8, 4, padding='SAME', activation_fn=tf.nn.relu)
            self.layer = tf.contrib.layers.conv2d(self.layer, 32, 4, 2, padding='SAME', activation_fn=tf.nn.relu)
            self.layer = tf.contrib.layers.conv2d(self.layer, 64, 4, 2, padding='SAME', activation_fn=tf.nn.relu)
            self.layer = tf.contrib.layers.flatten(self.layer)
            self.layer = tf.contrib.layers.fully_connected(self.layer, self.lstm_size, activation_fn=tf.nn.relu)
            self.layer = tf.reshape(self.layer, [1, tf.shape(self.state)[0], self.lstm_size])
    
            self.cells = [tf.contrib.rnn.LSTMCell(self.lstm_size) for _ in range(self.lstm_ncells)]
            self.multicell = tf.contrib.rnn.MultiRNNCell(self.cells)
            self.lstm_zero_state = self.multicell.zero_state(batch_size=1, dtype=tf.float32)

            self.initial_lstm_state = []
            for cellno in range(self.lstm_ncells):
                init_c = self.ph_lstm_state[cellno, 0, ...]
                init_h = self.ph_lstm_state[cellno, 1, ...]
                lstm_tuple = tf.nn.rnn_cell.LSTMStateTuple(init_c, init_h)
                self.initial_lstm_state.append(lstm_tuple)
            self.initial_lstm_state = tuple(self.initial_lstm_state)
    
            self.outputs, self.final_lstm_state = tf.nn.dynamic_rnn(self.multicell, self.layer, initial_state=self.initial_lstm_state, dtype=tf.float32)
    
            value_fn = lambda x: tf.contrib.layers.fully_connected(x, 1, activation_fn=None)
            policy_fn = lambda x: tf.contrib.layers.fully_connected(x, self.action_size, activation_fn=None)
    
            self.logits = tf.map_fn(policy_fn, self.outputs)
            self.policy = tf.nn.softmax(self.logits)
            self.actions_max = tf.argmax(self.logits, axis=1)
    
            self.value = tf.map_fn(value_fn, self.outputs)

            self.advantages = self.value_target - self.value
    
            # policy loss
            # the squeeze because maybe implementation changes to allow multibatch learning
            self.neglogpolicy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.squeeze(self.logits), labels=self.actions)
            # policy entropy
            self.policy_loss = tf.reduce_mean(tf.multiply(self.neglogpolicy, self.advantages))
            # value loss
            self.value_loss = tf.reduce_mean(tf.squared_difference(self.value, self.value_target))
            # entropy
            self.entropy = tf.reduce_mean(self.cat_entropy(self.logits))
            # combined loss
            self.loss = self.coeff_p * self.policy_loss + \
                        self.coeff_v * self.value_loss + \
                        self.coeff_h * self.entropy
        
            # train
            self.train_step = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.loss)

    def cat_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

    def set_gc_names(self):
        for k, v in self.__dict__.items():
            if isinstance(v, tf.Tensor) or isinstance(v, tf.Operation):
                self.gc[k] = self.__dict__[k].name
            if isinstance(v, list):
                if all(map(lambda x: isinstance(x, tf.Tensor) or isinstance(x, tf.Operation), v)):
                    self.gc[k] = [_.name for _ in v]
        
    def print_gc_names(self):
        for k, n in self.gc.items():
            print(k, n)
            
    def get_from_graph(self, name, graph):
        try:
            return graph.get_tensor_by_name(name)
        except:
            try:
                return graph.get_operation_by_name(name)
            except:
                raise 
           
    def init_from_graph(self, graph, gc):
        self.gc = gc
        for k, n in self.gc.items():
            if isinstance(n, list):
                self.__dict__[k] = [self.get_from_graph(_, graph) for _ in n]
            else:
                self.__dict__[k] = self.get_from_graph(n, graph)

def test_a3c():
	a3cnet = A3CNet()

def test_gradient():
    print('test_gradient')
    # build a small net
    with tf.variable_scope('test_network'):
        # input 
        x = tf.placeholder(dtype=tf.float32, shape=(None, 5))
        # trainable variable
        v = tf.Variable(tf.truncated_normal(shape=[5, 1], stddev=0.1))
        # function
        y = x @ v
        # compute gradients
        grads = tf.gradients(y, v)
        # optimizer
        trainer = tf.train.RMSPropOptimizer(learning_rate=1.0)
        # train step
        params = tf.trainable_variables()
        grads = tf.gradients(y, params)
        grads_and_vars = list(zip(grads, params))
        train = trainer.apply_gradients(grads_and_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        params = tf.trainable_variables()
        print('params = ', params) # params is list o variables, in this case [v]
        print('v = ', v)

        veval = v.eval()
        veval_old = veval
        # print('veval = ', veval)
        xval = np.array([[-2, -1, 0, 1, 2]])
        # yeval, gradeval = sess.run([y, grads], feed_dict={x: xval})
        # print('yeval = ', yeval)
        # print('gradeval = ', gradeval)

        _ = sess.run([grads, train], feed_dict={x: xval})
        grads = _[0][0]
        print(grads)

        veval = v.eval()
        diff = veval - veval_old
        print(diff)
        print('veval = ', np.concatenate([veval_old, veval, grads, diff], axis=1))

def test_lock_sync():
    pass

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

    def process_raw_state(self, state):
        self.push(self.process_state(state))
        return self.get()

class TestAgent(threading.Thread):
    def __init__(self, id, lock):
        threading.Thread.__init__(self)
        self.id = id
        self.lock = lock

    def short(self, it):
        print('ID{:2} it{:2} sentence1'.format(self.id, it))
        print('ID{:2} it{:2} sentence2'.format(self.id, it))

    def run(self):
        for _ in range(10):
            random_wait_time = np.random.uniform(1, 2)
            self.lock.acquire()
            self.short(_)
            self.lock.release()

def test_model_save_and_load():
    lock = threading.Lock()
    agents = [TestAgent(i, lock) for i in range(4)]

    for agent in agents:
        agent.start()

    while True:
        if all(not agent.isAlive() for agent in agents):
            break

    print('All done.')
    print('counter = {}'.format(counter))

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

class EnvBreakoutWrapper():

    def __init__(self):
        self.env = gym.make('Breakout-v0')
        self.state = None
        self.prev_state = None

    def reset(self):
        state = self.env.reset()
        self.state = self.process_im(state)

    def step(self, action):
        translated_action = action + 1
        next_state, reward, done, info = self.env.step(action)
        self.prev_state = self.state
        self.state = self.process_im(next_state)
        return reward, done, info

    def process_im(self, im):
        # crop
        im = im[32:192, :, :]
        # downsample
        im = im[::2, ::2, :]
        # normalize
        im = im/255.0
        # calculate luminance
        im = np.dot(im, [0.299, 0.587, 0.114])
        # add a channel dimension
        im = im[...,np.newaxis]
        return im

class A3CAgent(threading.Thread):
    def __init__(self, threadid, a3cnet, sess, args):
        threading.Thread.__init__(self)
        self.threadid = threadid
        self.a3cnet = a3cnet
        self.sess = sess
        self.args = args

    def run(self):
        args = self.args
        sess = self.sess

        seq_length = args.seq_length

        env = EnvBreakoutWrapper()
    
        states = np.zeros((seq_length, *self.a3cnet.state_shape), dtype=np.float32)
        actions = np.zeros((seq_length, ), dtype=np.int32)
        rewards = np.zeros((seq_length, ), dtype=np.float32)
        discounted_utilities = np.zeros((seq_length, ), dtype=np.float32)

        with sess.as_default():    
        # with tqdm(desc='Episode', total=args.episodes, unit=' episodes') as pbar:
            assert tf.get_default_session() is sess, 'session mismatch'
            # for episode in range(args.episodes):
            global episode

            while episode < args.episodes:
                total_rewards = 0
                episode_completed = False
                eps_idx = 0
                # reset environment
                env.reset()
                lstm_state = self.a3cnet.zero_lstm_state()
                while not episode_completed:
                    for seq_idx in range(seq_length):
                        # determine policy
                        feed_dict = {
                            self.a3cnet.state: env.state[np.newaxis,:], 
                            self.a3cnet.ph_lstm_state: lstm_state}
                        policy = self.a3cnet.policy.eval(feed_dict=feed_dict)
                        res = sess.run([self.a3cnet.policy, self.a3cnet.value, self.a3cnet.final_lstm_state], feed_dict=feed_dict)
                        policy = res[0][0][0] # [][batch][time]
                        value = res[1][0][0]
                        lstm_state = res[2]
                        assert all(policy>=0), "policy not >0 {}".format(policy)
                        assert abs(sum(policy)-1)<1e4, "sum policy not 1 {}".format(sum(policy))
                        # select action according to policy
                        action = np.random.choice(self.a3cnet.action_size, p=policy)
                        # execute action, get reward and next state
                        reward, done, info = env.step(action)
                        # update sum_of_rewards
                        total_rewards += reward
                        # save state, action, reward
                        states[seq_idx,:] = env.prev_state
                        actions[seq_idx] = action
                        rewards[seq_idx] = reward
                    
                        # end episode if done    
                        if done:
                            break
                    
                    seq_end = (seq_idx+1)
                    eps_idx += (seq_idx+1)

                    # calculate discounted utilities
                    if not done:
                        running_sum = value
                    else:
                        running_sum = 0.0
                    for reverse_idx in range(seq_idx, -1, -1):
                        running_sum += args.gamma * running_sum + rewards[reverse_idx]
                        discounted_utilities[reverse_idx] = running_sum

                    # train
                    feed_dict = {
                        self.a3cnet.state : states[:seq_end, ...],
                        self.a3cnet.actions : actions[:seq_end],
                        self.a3cnet.value_target : discounted_utilities[:seq_end, ...],
                        self.a3cnet.lr : args.lr,
                        self.a3cnet.coeff_p : args.coeff_p,
                        self.a3cnet.coeff_v : args.coeff_v,
                        self.a3cnet.coeff_h : args.coeff_h,
                        self.a3cnet.ph_lstm_state: self.a3cnet.zero_lstm_state()
                    }
                    run_result = sess.run([self.a3cnet.loss, self.a3cnet.train_step], 
                            feed_dict=feed_dict)

                    # determine whether episode completed
                    if done or eps_idx > args.max_episode_length:
                        episode_completed = True

                # save sum_of_rewards
                #scores.append(total_rewards)
                # episode += 1        

def test_a3cagentssimulation():
    args = arguments()

    a3cnet = A3CNet()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    agents = [A3CAgent(i, a3cnet, sess, args) for i in range(1)]

    for agent in agents:
        agent.start()

    while True:
        if all(not agent.isAlive() for agent in agents):
            break

    print('All done.')        

def arguments():
    parser = argparse.ArgumentParser(description='A3C Cartpole')
    parser.add_argument('--episodes', type=int, default=2000, metavar='E',
                        help='number of training episodes (default: 2000)')
    parser.add_argument('--seq-length', type=int, default=4, metavar='L',
                        help='sequence length (default: 4)')     
    parser.add_argument('--max-episode-length', type=int, default=500, metavar='L',
                        help='maximum episode length (default: 500)')    
    parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--coeff-p', type=float, default=1.0, metavar='cp',
                        help='loss coefficient policy (default: 1.0)')
    parser.add_argument('--coeff-v', type=float, default=0.5, metavar='cv',
                        help='loss coefficient value (default: 0.5)')
    parser.add_argument('--coeff-h', type=float, default=0.01, metavar='ch',
                        help='loss coefficient entropy (default: 0.01)')
    parser.add_argument('--number-of-agents', type=int, default=4, metavar='na',
                        help='number of agents (default: 4)')

    args = parser.parse_args()
    return args

def test_recursive_gc():
    # check that all elements are strings
    x = []

    def recursive_gc(obj):

        if isinstance(obj, str):
            return True, len(obj)
        elif isinstance(obj, list):
            truths, res = zip(*[recursive_gc(_) for _ in obj])
            return all(truths), res
        elif isinstance(obj, tuple):
            truths, res = zip(*[recursive_gc(_) for _ in obj])
            return all(truths), tuple(res)
        else:
            return False, None

    print(recursive_gc('abc'))
    print(recursive_gc(['abc', 'xy']))
    print(recursive_gc(['abc', ['xy', 'd', 'e']]))
    print(recursive_gc(['abc', ['xy', 'd', None]]))
    print(recursive_gc(['abc', ('xy', 'd', ['a', 'b'])]))

def main():
    # test_a3c()
    # test_gradient()
    # test_model_save_and_load()
    #test_a3cagentssimulation()
    test_recursive_gc()

if __name__ == '__main__':
    main()
