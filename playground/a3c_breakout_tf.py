import gym
import tensorflow as tf

import time
import threading
import argparse
import pickle
import numpy as np
from tqdm import tqdm

import pylab
import matplotlib.pyplot as plt

episode = 0
scores = []
stats = {
    'policy_loss': [],
    'value_loss': [],
    'entropy': [],
}

class A3CNet():

    def __init__(self, state_shape=(80, 80, 1), action_size=3, 
                seq_batch_size=32, seq_len=8, lstm_size=128, lstm_ncells=2,
                alpha=0.2, beta1=0.2, smooth=0.9, max_grad_norm=0.05,
                build=True):
        self.name = 'A3CNet'

        self.state_shape = state_shape
        self.action_size = action_size

        self.alpha = alpha
        self.beta1 = beta1
        self.smooth = smooth
        self.max_grad_norm = max_grad_norm
        
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

    def multiArrayToLSTMStateTuple(self, marray):
        lstm_state = []
        for cellno in range(self.lstm_ncells):
            init_c = self.ph_lstm_state[cellno, 0, ...]
            init_h = self.ph_lstm_state[cellno, 1, ...]
            lstm_tuple = tf.nn.rnn_cell.LSTMStateTuple(init_c, init_h)
            lstm_state.append(lstm_tuple)
        lstm_state = tuple(lstm_state)
        return lstm_state

    def LSTMStateTupleToMultiArray(self, lstm_state):
        marray = []
        for cellno in range(self.lstm_ncells):
            marray.append(tuple([lstm_state[cellno].c, lstm_state[cellno].h]))
        return marray

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
    
            self.outputs, final_lstm_state_ = tf.nn.dynamic_rnn(
                self.multicell, self.layer, initial_state=self.multiArrayToLSTMStateTuple(self.ph_lstm_state), dtype=tf.float32)
            self.final_lstm_state = self.LSTMStateTupleToMultiArray(final_lstm_state_)

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
            self.H = self.cat_entropy(self.logits)
            self.entropy = tf.reduce_mean(self.cat_entropy(self.logits))
            # combined loss
            self.loss = self.coeff_p * self.policy_loss + \
                        self.coeff_v * self.value_loss - \
                        self.coeff_h * self.entropy # for exploration we want to increase entropy
        
            # train
            # self.train_step = tf.train.AdamOptimizer(self.lr, beta1=self.beta1).minimize(self.loss)
            self.trainer = tf.train.RMSPropOptimizer(self.lr)
            # train step
            self.params = tf.trainable_variables()
            self.grads = tf.gradients(self.loss, self.params)
            if self.max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(self.grads, self.max_grad_norm)
            self.grads_and_vars = list(zip(self.grads, self.params))
            self.train_step = self.trainer.apply_gradients(self.grads_and_vars)

    def cat_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, -1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, -1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), -1)

    def set_gc_names(self):

        def recursive_gc(obj):
            if isinstance(obj, tf.Tensor) or isinstance(obj, tf.Operation):
                return True, obj.name
            elif isinstance(obj, list):
                truths, res = zip(*[recursive_gc(_) for _ in obj])
                return all(truths), res
            elif isinstance(obj, tuple):
                truths, res = zip(*[recursive_gc(_) for _ in obj])
                return all(truths), tuple(res)
            else: return False, None

        for k, v in self.__dict__.items():
            t, res = recursive_gc(v)
            if t:
                self.gc[k] = res
        
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

        def recursive_init(obj, graph):
            if isinstance(obj, list):
                return [recursive_init(_, graph) for _ in obj]
            if isinstance(obj, tuple):
                return tuple(recursive_init(_, graph) for _ in obj)              
            else:
                return self.get_from_graph(obj, graph)

        self.gc = gc
        for k, n in self.gc.items():
            self.__dict__[k] = recursive_init(n, graph)

class EnvBreakoutWrapper():

    def __init__(self, reset_noop=30, action_repeat=4):
        self.env = gym.make('BreakoutNoFrameskip-v4') #('Breakout-v0')
        self.state = None
        self.prev_state = None
        self.reset_noop = reset_noop
        self.action_repeat = action_repeat

    def reset(self):
        state = self.env.reset()
        self.state = self.process_im(state)
        for i in range(self.reset_noop):
            next_state, reward, done, info = self.env.step(0)
            self.prev_state = self.state
            self.state = self.process_im(next_state)

    def step(self, action):
        reward = 0
        translated_action = action + 1
        for i in range(self.action_repeat):
            next_state, reward_, done, info = self.env.step(translated_action)
            reward += reward_
            self.prev_state = self.state
            self.state = self.process_im(next_state)
            if done:
                break
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

    def render(self):
        self.env.render()

    def next_life(self):
        next_state, reward, done, info = self.env.step(1)

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
        values = np.zeros((seq_length, ), dtype=np.float32)
        discounted_utilities = np.zeros((seq_length, ), dtype=np.float32)

        with sess.as_default():    
        # with tqdm(desc='Episode', total=args.episodes, unit=' episodes') as pbar:
            assert tf.get_default_session() is sess, 'session mismatch'
            # for episode in range(args.episodes):
            global episode
            global scores
            global stats

            while episode < args.episodes:
                total_rewards = 0
                episode_completed = False
                eps_idx = 0
                lives = 5
                # reset environment
                env.reset()
                env.next_life()
                lstm_state = self.a3cnet.zero_lstm_state()
                while not episode_completed:
                    # gather loss statistics
                    policy_loss_list, value_loss_list, entropy_list = [], [], []
                    # save the LSTM state at the beginning of a sequence
                    lstm_state_seq_start = lstm_state
                    # simulate a sequence of steps
                    for seq_idx in range(seq_length):
                        # determine policy
                        feed_dict = {
                            self.a3cnet.state: env.state[np.newaxis,:], 
                            self.a3cnet.ph_lstm_state: lstm_state}
                        res = sess.run([self.a3cnet.policy, self.a3cnet.value, self.a3cnet.final_lstm_state], feed_dict=feed_dict)
                        policy = res[0][0][0]   # [][batch][time]
                        value = res[1][0][0][0] # [][batch][time][]
                        lstm_state = res[2]
                        assert all(policy>=0), "policy not >0 {}".format(policy)
                        assert abs(sum(policy)-1)<1e4, "sum policy not 1 {}".format(sum(policy))
                        # select action according to policy
                        action = np.random.choice(self.a3cnet.action_size, p=policy)
                        # execute action, get reward and next state
                        reward, done, info = env.step(action)
                        # save state, action, reward
                        states[seq_idx,:] = env.prev_state
                        actions[seq_idx] = action
                        rewards[seq_idx] = reward                        
                        values[seq_idx] = value                        
                        # check if dead
                        if lives > info['ale.lives']: # has died
                            lives = info['ale.lives']
                            reward = -1
                            env.next_life()
                        # update sum_of_rewards
                        total_rewards += reward
                        # end episode if done    
                        if done:
                            break
                    
                    seq_end = (seq_idx+1)
                    eps_idx += (seq_idx+1)

                    # calculate discounted utilities
                    if not done:
                        feed_dict = {
                            self.a3cnet.state: env.state[np.newaxis,:], 
                            self.a3cnet.ph_lstm_state: lstm_state}
                        res = sess.run(self.a3cnet.value, feed_dict=feed_dict)
                        value = res[0]             
                        running_sum = value
                    elif seq_idx > 0:
                        seq_end = seq_idx
                        running_sum = value
                    else:
                        episode_completed = True
                        break # nothing to train on
                    for reverse_idx in range(seq_idx, -1, -1):
                        running_sum = args.gamma * running_sum + rewards[reverse_idx]
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
                        self.a3cnet.ph_lstm_state: lstm_state_seq_start
                    }
                    run_result = sess.run([self.a3cnet.loss, self.a3cnet.train_step, 
                                           self.a3cnet.policy_loss, self.a3cnet.value_loss, self.a3cnet.entropy,
                                           self.a3cnet.value, self.a3cnet.value_target, 
                                           self.a3cnet.policy, self.a3cnet.logits, self.a3cnet.H],
                            feed_dict=feed_dict)

                    loss, policy_loss, value_loss, entropy = run_result[0], run_result[2], run_result[3], run_result[4]
                    value, value_target = run_result[5], run_result[6]
                    policy = run_result[7]
                    logits = run_result[8]
                    H = run_result[9]
                    policy_loss_list.append(policy_loss)
                    value_loss_list.append(value_loss)
                    entropy_list.append(entropy)
                    if (eps_idx // seq_length) % 32 == 0:
                        print('policy_loss={:2f}, value_loss={:2f}, entropy={:2f}, loss={:2f} '.format(policy_loss, value_loss, entropy, loss), end='')
                        print(actions[:seq_end])
                        # print(policy[0][0])
                        # print(logits[0][0])
                        # print(H)
                    # print(rewards[:seq_end])
                    # print(discounted_utilities[:seq_end])
                    # print(values[:seq_end])
                    
                    # print('value ', value)
                    # print('value_target ', value_target)

                    # determine whether episode completed
                    if done or eps_idx > args.max_episode_length:
                        episode_completed = True

                # save sum_of_rewards
                scores.append(total_rewards)
                stats['policy_loss'].append(np.mean(policy_loss_list) if len(policy_loss_list) > 0 else 0)
                stats['value_loss'].append(np.mean(value_loss_list) if len(value_loss_list) > 0 else 0)
                stats['entropy'].append(np.mean(entropy_list) if len(entropy_list) > 0 else 0)
                episode += 1        

def plot_results(fig, axes, scores, policy_loss, value_loss, entropy, save=None):
    axes[0][0].set_title('Scores')
    axes[0][0].plot(scores, 'b')
    axes[0][1].set_title('Policy loss')
    axes[0][1].plot(policy_loss, 'g-')
    axes[1][0].set_title('Value loss')
    axes[1][0].plot(value_loss, 'r-')
    axes[1][1].set_title('Entropy')
    axes[1][1].plot(entropy, 'm-')

    if not save is None:
        plt.savefig(save)

def test_a3c():
    args = arguments()

    if args.build:
        print('Building model ...')
        model = A3CNet()
        # save graph config 
        with open('save_model/' + model.name + '.pickle', 'wb') as f:
            pickle.dump(model.gc, f)
    else:
        print('Loading model ...')
        model = A3CNet(build=False)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    if args.build:
        saver = tf.train.Saver()
    else:
        saver = tf.train.import_meta_graph('save_model/' + model.name + '.meta')
        saver.restore(sess, 'save_model/' + model.name)
        with open('save_model/' + model.name + '.pickle', 'rb') as f:
            gc = pickle.load(f) 
        model.init_from_graph(sess.graph, gc)

    agents = [A3CAgent(i, model, sess, args) for i in range(args.number_of_agents)]

    for agent in agents:
        time.sleep(0.05)
        agent.start()

    # Create figure for plotting
    ncols, nrows = 2, 2
    fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(4.5*ncols,4.5*nrows))

    while True:
        time.sleep(1)

        plot_results(fig, axes, scores[:], stats['policy_loss'][:], stats['value_loss'][:], stats['entropy'][:], save="./save_graph/a3c_breakout_tf.png")

        if all(not agent.isAlive() for agent in agents):
            break

    # save the model
    save_path = saver.save(sess, 'save_model/' + model.name)
    print('All done.')

def play_saved_model():
    print('Executing play_saved_model ...')
    args = arguments()

    model = A3CNet(build=False)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.import_meta_graph('save_model/' + model.name + '.meta')
    saver.restore(sess, 'save_model/' + model.name)
    with open('save_model/' + model.name + '.pickle', 'rb') as f:
        gc = pickle.load(f) 
    model.init_from_graph(sess.graph, gc)

    env = EnvBreakoutWrapper()
    
    with sess.as_default():    
        assert tf.get_default_session() is sess, 'session mismatch'
        
        env.reset()
        env.next_life()
        lstm_state = model.zero_lstm_state()
        lives = 5
        step = 0
        total_rewards = 0
        done = False
        while not done:
            env.render()

            feed_dict = {
                model.state: env.state[np.newaxis,:], 
                model.ph_lstm_state: lstm_state}
            res = sess.run([model.policy, model.value, model.final_lstm_state], feed_dict=feed_dict)
            policy = res[0][0][0] # [][batch][time]
            lstm_state = res[2]
            action = np.random.choice(model.action_size, p=policy)
            # execute action, get reward and next state
            reward, done, info = env.step(action)            

            total_rewards += reward
            print('policy ', policy)
            print('{} action = {}, reward = {}, total = {}, lives = {}'.format(step, action, reward, total_rewards, info['ale.lives']))
            step += 1
            time.sleep(0.03)

            if lives > info['ale.lives']: # has dies
                lives = info['ale.lives']
                env.next_life()

            input()

    print('All done.')

def arguments():

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def fsel(v):
        if v.lower() in ('train', 'true', 'y', '1', 't'):
            return True
        else:
            return False

    parser = argparse.ArgumentParser(description='A3C Cartpole')
    parser.add_argument('--build', type=str2bool, default=False, metavar='B',
                        help='build or restore model (default: False)')     
    parser.add_argument('--train', type=fsel, default=False, metavar='B',
                        help='train (True) or play (False) (default: False)')    
    parser.add_argument('--episodes', type=int, default=2000, metavar='E',
                        help='number of training episodes (default: 2000)')
    parser.add_argument('--seq-length', type=int, default=16, metavar='L',
                        help='sequence length (default: 16)')     
    parser.add_argument('--max-episode-length', type=int, default=10000, metavar='L',
                        help='maximum episode length (default: 10000)')    
    parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='lr',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--coeff-p', type=float, default=1.0, metavar='cp',
                        help='loss coefficient policy (default: 1.0)')
    parser.add_argument('--coeff-v', type=float, default=0.5, metavar='cv',
                        help='loss coefficient value (default: 0.5)')
    parser.add_argument('--coeff-h', type=float, default=0.01, metavar='ch',
                        help='loss coefficient entropy (default: 0.01)')
    parser.add_argument('--number-of-agents', type=int, default=1, metavar='na',
                        help='number of agents (default: 1)')

    args = parser.parse_args()
    return args

def test_entropy():

    def entropy(logits):
        a0 = logits - np.max(logits)
        print('a0=', a0)
        ea0 = np.exp(a0)
        print('ea0=', ea0)
        z0 = np.sum(ea0)
        print('z0=', z0)
        p0 = ea0 / z0
        print('p0=', p0)
        print('np.log(z0)=', np.log(z0))
        print('np.log(z0)-a0=', np.log(z0)-a0)
        print('p0 * (np.log(z0) - a0)=', p0 * (np.log(z0) - a0))
        return np.sum(p0 * (np.log(z0) - a0))

    # print(entropy([10,1,1]))
    # print(entropy([2,2,2]))
    print(entropy([0.99,0.005,0.005]))
    t = np.array([0.99,0.005,0.005])
    p = np.exp(t)/np.sum(np.exp(t))
    print('p=', p)
    print('-log(p)=', -np.log(p))
    print(-np.sum(p * np.log(p)))


def small_test():
    env = gym.make('BreakoutNoFrameskip-v4')
    print(env.unwrapped.get_action_meanings())

def main():
    args = arguments()
    if args.train:
        test_a3c()
    else:
        play_saved_model()

if __name__ == '__main__':
    main()
    # test_entropy()
    # small_test()
