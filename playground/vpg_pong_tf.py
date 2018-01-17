# Vanilla Policy Gradient for Pong
import gym
import tensorflow as tf
import sys
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from getchar import _Getch

class EnvWrapperPong():

    def __init__(self, reset_noop=30, action_repeat=1):
        self.env = gym.make('Pong-v0') #('Breakout-v0')
        self.state = None
        self.prev_state = None
        self.diff_state = None
        self.reset_noop = reset_noop
        self.action_repeat = action_repeat

    def reset(self):
        state = self.env.reset()
        self.state = self.process_im(state)
        for i in range(self.reset_noop):
            next_state, reward, done, info = self.env.step(0)
            self.prev_state = self.state
            self.state = self.process_im(next_state)
            self.diff_state = self.state - self.prev_state

    def action_decode(self, action_code):
    	code = {
    	    0: 'u',
    	    1: 'd',
    	    2: '_'
    	}
    	return code.get(action_code, '_')

    def step(self, action):
        # assert action in ['u', 'd'], "Action is not up or down."
        action = self.action_decode(action) if isinstance(action, int) else action
        reward = 0
        if action.lower() == 'u':
            translated_action = 2
        elif action.lower() == 'd':
            translated_action = 3
        else:
            translated_action = 1
        for i in range(self.action_repeat):
            next_state, reward_, done, info = self.env.step(translated_action)
            reward += reward_
            self.prev_state = self.state
            self.state = self.process_im(next_state)
            self.diff_state = self.state - 0.5 * self.prev_state
            if done:
                break
        return reward, done, info

    def process_im_(self, im):
        # crop
        im = im[34:194, :, :]
        # downsample
        im = im[::2, ::2, :]
        # normalize
        im = im/255.0
        # calculate luminance
        im = np.dot(im, [0.299, 0.587, 0.114])
        # add a channel dimension
        im = im[...,np.newaxis]
        return im

    def process_im(self, im):
        # crop
        im = im[34:194, :, :]
        # downsample, select R channel
        im = im[::2, ::2, 0]
        im[im == 144] = 0 # erase background (background type 1)
        im[im == 109] = 0 # erase background (background type 2)
        im[im != 0] = 1 # everything else (paddles, ball) just set to 1        
        # normalize
        im = im/255.0
        # add a channel dimension
        im = im[...,np.newaxis]
        return im

    def render(self):
        self.env.render()

    def next_life(self):
        next_state, reward, done, info = self.env.step(1)

class Net():

    def __init__(self, build=True, save_path='save_model'):
        self.save_path = save_path if save_path[-1] == '/' else save_path + '/'

        self.saver = None
        # graph config
        self.gc = dict()
        
        if build:
            self.build()
            # Save names
            self.set_gc_names()
            # save graph config 
            with open(self.save_path + self.name + '.pickle', 'wb') as f:
                pickle.dump(self.gc, f)

    def build(self):
        raise NotImplemented

    def init_saver(self, sess, build):
        if build:
            self.saver = tf.train.Saver()
        else:
            self.saver = tf.train.import_meta_graph(self.save_path + self.name + '.meta')
            self.saver.restore(sess, self.save_path + self.name)
            with open(self.save_path + self.name + '.pickle', 'rb') as f:
                gc = pickle.load(f) 
            self.init_from_graph(sess.graph, gc)

    def save_model(self, sess):
        # save the model
        save_path = self.saver.save(sess, self.save_path + self.name)

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



class VPGNet(Net):

    def __init__(self, 
                state_shape=(80, 80, 1), action_size=3, 
                build=True, save_path='save_model'):

        self.name = 'VPGNet'

        self.state_shape = state_shape
        self.action_size = action_size

        super(VPGNet, self).__init__(build, save_path)

    def placeholders(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, *self.state_shape))
        self.actions = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, ))
        self.lr = tf.placeholder(dtype=tf.float32)
        self.rmsprop_decay = tf.placeholder(dtype=tf.float32)
        self.policy_coef = tf.placeholder(dtype=tf.float32)
        self.entropy_coef = tf.placeholder(dtype=tf.float32)  

    def build(self):
        self.placeholders()

        with tf.variable_scope(self.name):
            self.layer = tf.contrib.layers.conv2d(self.state, 8, 5, 2, padding='SAME', activation_fn=tf.nn.relu)
            print(self.layer.get_shape())
            self.layer = tf.contrib.layers.conv2d(self.layer, 16, 5, 2, padding='SAME', activation_fn=tf.nn.relu)
            print(self.layer.get_shape())
            self.layer = tf.contrib.layers.conv2d(self.layer, 16, 5, 2, padding='SAME', activation_fn=tf.nn.relu)
            print(self.layer.get_shape())
            self.layer = tf.contrib.layers.conv2d(self.layer, 16, 5, 2, padding='SAME', activation_fn=tf.nn.relu)
            print(self.layer.get_shape())            
            self.layer = tf.contrib.layers.flatten(self.layer)
            print(self.layer.get_shape())
            self.layer = tf.contrib.layers.fully_connected(self.layer, 32, activation_fn=tf.nn.relu)
            print(self.layer.get_shape())
            self.logits = tf.contrib.layers.fully_connected(self.layer, self.action_size, activation_fn=None)
            self.policy = tf.nn.softmax(self.logits)

            self.entropy = self.cat_entropy(self.logits)
            self.mean_entropy = tf.reduce_mean(self.entropy)

            self.logp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.logits)
            self.policy_loss = tf.reduce_mean(
                            tf.multiply(self.advantage, self.logp)
                        )

            self.loss = self.policy_loss * self.policy_coef - \
                        self.mean_entropy * self.entropy_coef
            self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=self.rmsprop_decay).minimize(self.loss)

    def cat_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, -1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, -1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), -1)

class VPG2Net(Net):

    def __init__(self, 
                state_shape=(80, 80, 1), action_size=1, 
                build=True, save_path='save_model'):

        self.name = 'VPG2Net'

        self.state_shape = state_shape
        self.action_size = action_size

        super(VPG2Net, self).__init__(build, save_path)

    def placeholders(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, *self.state_shape))
        self.action = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, ))
        self.lr = tf.placeholder(dtype=tf.float32)
        self.policy_coef = tf.placeholder(dtype=tf.float32)
        self.entropy_coef = tf.placeholder(dtype=tf.float32)  

    def build(self):
        self.placeholders()

        with tf.variable_scope(self.name):
            self.layer = tf.contrib.layers.conv2d(self.state, 8, 8, 4, padding='SAME', activation_fn=tf.nn.relu)
            print(self.layer.get_shape())
            self.layer = tf.contrib.layers.conv2d(self.layer, 8, 5, 2, padding='SAME', activation_fn=tf.nn.relu)
            print(self.layer.get_shape())
            self.layer = tf.contrib.layers.conv2d(self.layer, 8, 5, 2, padding='SAME', activation_fn=tf.nn.relu)
            print(self.layer.get_shape())
            self.layer = tf.contrib.layers.flatten(self.layer)
            print(self.layer.get_shape())
            self.layer = tf.contrib.layers.fully_connected(self.layer, 8, activation_fn=tf.nn.relu)
            print(self.layer.get_shape())
            self.logits = tf.contrib.layers.fully_connected(self.layer, self.action_size, activation_fn=None)
            self.policy = tf.nn.sigmoid(self.logits)

            self.entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.policy, logits=self.logits)
            self.mean_entropy = tf.reduce_mean(self.entropy)

            self.logp = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.action, logits=self.logits)
            self.policy_loss = tf.reduce_mean(
                            tf.multiply(self.logp, self.advantage)
                        )

            self.loss = self.policy_loss * self.policy_coef - \
                        self.mean_entropy * self.entropy_coef
            self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def cat_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, -1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, -1, keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), -1)


def user_pong(args):
    # variables
    getcharobj = _Getch()
    reward, done, info = None, False, None
    # initialize environment
    env = gym.make('Pong-v0')
    state = env.reset()

    print(env.unwrapped.get_action_meanings())

    while not done:
        env.render()
        
        x = getcharobj()
        
        if x.lower() == 'u':
            action = 2
        elif x.lower() == 'd':
            action = 3
        else:
            action = 1

        state, reward, done, info = env.step(action)

def user_pong_w_processing(args):
    # variables
    fig = plt.figure()
    getcharobj = _Getch()
    reward, done, info = None, False, None
    # initialize environment
    env = EnvWrapperPong()
    state = env.reset()

    while not done:
        display_vec = (np.squeeze(env.diff_state) + 1)/2.0
        plt.imshow(display_vec, cmap='gray')
        plt.savefig("./save_graph/vpg_pong_tf_gameplay.png")
        
        x = getcharobj()
        reward, done, info = env.step(x)

def pg_alg(args):

    def savefig(fig, axes, scores, lengths, losses, entropies, fname=None):
        axes[0][0].cla()
        axes[0][1].cla()
        axes[1][0].cla()
        axes[1][1].cla()

        axes[0][0].set_title('Scores')
        axes[0][0].plot(scores, 'b')
        axes[0][1].set_title('Length')
        axes[0][1].plot(lengths, 'g-')
        axes[1][0].set_title('Value loss')
        axes[1][0].plot(losses, 'r-')
        axes[1][1].set_title('Entropies')
        axes[1][1].plot(entropies, 'm-')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)       
    
        if not fname is None:
            plt.savefig(fname)

    def discounted_rewards(rewards, start_idx, end_idx):
        gamma = None
        gamma_l = 0.97
        positive = True
        running_sum = 0.0
        running_sum_l = 0.0
        
        eps_start = -1
        eps_end = -1
        eps_len = -1


        for idx in range(end_idx-1, start_idx-1, -1):
            if rewards[idx] == 0:
                running_sum = gamma * running_sum
                running_sum_l = gamma_l * running_sum_l

                if idx == 0 or rewards[idx-1] != 0:
                    eps_start = idx
                    if eps_end > eps_start:
                        eps_len = eps_end - eps_start + 1
                        if eps_len > 60:
                            rewards[eps_start+40:eps_start+55] += 1.0
            else:
                eps_end = idx
                
                running_sum = rewards[idx]
                if rewards[idx] > 0:
                    positive = True
                    gamma = 0.9999
                elif rewards[idx] < 0:
                    positive = False
                    gamma = 0.93

            if positive:
                rewards[idx] = running_sum
            else:
                if running_sum < -0.01:
                    rewards[idx] = running_sum
                    running_sum_l = 0.5
                else:
                	rewards[idx] = running_sum + 0.5 - running_sum_l            	

        # rewards[start_idx:end_idx] -= np.mean(rewards[start_idx:end_idx])
        # rewards[start_idx:end_idx] /= np.std(rewards[start_idx:end_idx])

    # figure for plotting learning process
    fig, axes = plt.subplots(2, 2, figsize=(3.0*2,3.0*2))

    env = EnvWrapperPong()

    losses = np.zeros((args.episodes))
    episode_lengths = np.zeros((args.episodes))
    entropies = np.zeros((args.episodes))
    total_rewards = np.zeros((args.episodes))

    state_shape = (80, 80, 1)
    action_size = 3
    model = VPGNet(state_shape=state_shape, action_size=3, build=args.build)

    state = env.reset()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # init saver
    model.init_saver(sess, args.build)

    # learn the policy probability distribution
    memsize = args.batch_size * args.max_episode_length
    states = np.zeros((memsize, *state_shape), dtype=np.float32)
    probs = np.zeros((memsize, action_size), dtype=np.int32)
    actions = np.zeros((memsize, ), dtype=np.int32)
    rewards = np.zeros((memsize, ), dtype=np.float32)
 
    with tqdm(desc='Episode', total=args.episodes, unit=' episodes') as pbar:
        batch_idx = 0
        batch_start = 0 	
        for episode in range(args.episodes):
            match_idx = 0
            for idx in range(args.max_episode_length):
                # determine policy
                policy = model.policy.eval(feed_dict={model.state: env.diff_state[np.newaxis,:]})[0]
                # select action according to policy
                action = np.random.choice(action_size, p=policy)
                # execute action, get reward and next state
                reward, done, info = env.step(action)
                # update total reward for episode
                total_rewards[episode] += reward
                # save state, action, reward
                step_idx = batch_start+idx
                states[step_idx,:] = env.diff_state
                probs[step_idx,:] = policy
                actions[step_idx] = action
                # add reward when match is long
                if reward != 0: # match ended
                    match_idx = 0
                else:
                    match_idx += 1
                    # if match_idx > 60: # add a reward at time 50, if player hits ball
                    #     rewards[step_idx-10] = 1
                rewards[step_idx] = reward
                # end episode if done                
                if done:
                    break

            # compute discounted utilities
            eps_len = idx+1
            batch_end = batch_start+eps_len
            discounted_rewards(rewards, batch_start, batch_end)

            feed_dict = {
                model.state: states[batch_start:batch_end,:],
                model.actions: actions[batch_start:batch_end],
                model.advantage: rewards[batch_start:batch_end], 
                model.policy_coef: args.coeff_p,
                model.entropy_coef: args.coeff_h
            }
            run_result = sess.run([model.loss, model.mean_entropy, model.entropy, model.policy], feed_dict=feed_dict)
            loss = run_result[0]
            entropy = run_result[1]
            losses[episode] = loss
            entropies[episode] = entropy
            episode_lengths[episode] = eps_len
            entropy = run_result[2]
            policy = run_result[3]

            # update output
            pbar.set_description('Episode {:>5}'.format(episode))
            pbar.set_postfix(Loss='{:>9.2f}'.format(loss), Reward='{:<5.1f}'.format(total_rewards[episode]))
            pbar.update(1)

            # save fig
            savefig(fig, axes, 
            	total_rewards[:episode+1], episode_lengths[:episode+1], losses[:episode+1], entropies[:episode+1], 
            	fname='./save_graph/vpg_pong_tf.png')

            # compute batch indices
            batch_idx = batch_idx + 1
            batch_start = batch_end
            if batch_idx == args.batch_size:
                # train 
                feed_dict = {
                    model.state: states[:batch_end,:],
                    model.actions: actions[:batch_end],
                    model.advantage: rewards[:batch_end], 
                    model.lr: args.learning_rate,
                    model.rmsprop_decay: args.rmsprop_decay,
                    model.policy_coef: args.coeff_p,
                    model.entropy_coef: args.coeff_h
                }
                run_result = sess.run([model.train_step], feed_dict=feed_dict)
                # update indices
                batch_idx = 0
                batch_start = 0

            # reset environment
            state = env.reset()            	
            

    model.save_model(sess)

def play(args):
    print('Play ...')

    state_shape = (80, 80, 1)
    action_size = 3
    model = VPGNet(state_shape=state_shape, action_size=3, build=False)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # init saver
    model.init_saver(sess, build=False)

    env = EnvWrapperPong()
    
    with sess.as_default():    
        assert tf.get_default_session() is sess, 'session mismatch'
        
        env.reset()
        step = 0
        total_rewards = 0
        done = False
        while not done:
            env.render()

            # determine policy
            policy = model.policy.eval(feed_dict={model.state: env.diff_state[np.newaxis,:]})[0]
            # select action according to policy
            action = np.random.choice(action_size, p=policy)
            # execute action, get reward and next state
            reward, done, info = env.step(action)

            if step > 48 and step < 53:
            	print(step)
            if reward == 0:
            	step += 1
            else:
            	print('Episode end at {} with reward {}'.format(step, reward))
            	step = 0
            total_rewards += reward
            time.sleep(0.02)

            # input()
    print('Total rewards {}'.format(total_rewards))
    print('All done.')    

def test_vpgnet(args):
    model = VPGNet()

def test_vpg2net(args):
    model = VPG2Net()

def test_rewards():

    def discounted_rewards(rewards, start_idx, end_idx):
        gamma = None
        gamma_l = 0.97
        positive = True
        running_sum = 0.0
        running_sum_l = 0.0
        
        eps_start = -1
        eps_end = -1
        eps_len = -1


        for idx in range(end_idx-1, start_idx-1, -1):
            if rewards[idx] == 0:
                running_sum = gamma * running_sum
                running_sum_l = gamma_l * running_sum_l

                if idx == 0 or rewards[idx-1] != 0:
                    eps_start = idx
                    if eps_end > eps_start:
                        eps_len = eps_end - eps_start + 1
                        if eps_len > 60:
                            rewards[eps_start+40:eps_start+55] += 1.0
            else:
                eps_end = idx
                
                running_sum = rewards[idx]
                if rewards[idx] > 0:
                    positive = True
                    gamma = 0.9999
                elif rewards[idx] < 0:
                    positive = False
                    gamma = 0.93

            if positive:
                rewards[idx] = running_sum
            else:
                if running_sum < -0.01:
                    rewards[idx] = running_sum
                    running_sum_l = 0.5
                else:
                	rewards[idx] = running_sum + 0.5 - running_sum_l

    def discounted_rewards_(rewards, start_idx, end_idx):
        gamma = None
        positive = True
        running_sum = 0.0
        eps_len = end_idx - start_idx

        if eps_len > 65:
        	rewards[:] = 0.0
        	rewards[40:55] = 1
        else:
        	rewards[:] = -1      	
        
        # print(np.mean(rewards[start_idx:end_idx]), np.std(rewards[start_idx:end_idx]))
        # rewards[start_idx:end_idx] -= np.mean(rewards[start_idx:end_idx])
        # rewards[start_idx:end_idx] /= np.std(rewards[start_idx:end_idx])

    # figure for plotting learning process
    fig, axes = plt.subplots(4, 1, figsize=(3.0*2,3.0*2))

    eps = [
        [(+1.0, 50), (+1.0, 85), (+1.0, 125), (+1.0, 200)],
        [(-1.0, 50), (-1.0, 85), (-1.0, 125), (-1.0, 200)],
        [(-1.0, 50), (-1.0, 85)],
        [(-1.0, 125), (-1.0, 260)]
    ]
    

    for i, e in enumerate(eps):    	
        elen = sum([t[1] for t in e])

        idx = 0
        rewards = np.zeros((elen, ), dtype=np.float32)
        for t in e:
            idx += t[1]
            rewards[idx-1] = t[0]

        discounted_rewards(rewards, 0, elen)
        axes[i].plot(rewards)
        axes[i].grid()
    plt.show()

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
	
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient')
    parser.add_argument('--build', type=str2bool, default=False, metavar='B',
                        help='build or restore model (default: False)')     
    parser.add_argument('--train', type=fsel, default=False, metavar='B',
                        help='train (True) or play (False) (default: False)')
    parser.add_argument('--net', type=int, default=2, metavar='N',
                        help='net model (default: 2)')    
    parser.add_argument('--episodes', type=int, default=1000, metavar='E',
                        help='number of training episodes (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='NB',
                        help='batch size (default: 1)')    
    parser.add_argument('--max-episode-length', type=int, default=3000, metavar='L',
                        help='maximum episode length (default: 3000)')
    parser.add_argument('--gamma_p', type=float, default=0.9999, metavar='gp',
                        help='discount factor (default: 0.9999)')
    parser.add_argument('--gamma_m', type=float, default=0.95, metavar='gp',
                        help='discount factor (default: 0.95)')    
    parser.add_argument('--rmsprop-decay', type=float, default=0.999, metavar='d',
                        help='RMS prop decay factor (default: 0.99)')    
    parser.add_argument('--learning-rate', type=float, default=0.002, metavar='lr',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--coeff-p', type=float, default=1.0, metavar='cp',
                        help='loss coefficient policy (default: 1.0)')
    parser.add_argument('--coeff-h', type=float, default=0.01, metavar='ch',
                        help='loss coefficient entropy (default: 0.01)')    

    args = parser.parse_args()
    return args    

if __name__ == '__main__':
    args = arguments()
    # user_pong_w_processing(args)
    # test_vpgnet(args)
    # test_vpg2net(args)
    # test_rewards()

    if args.train:
        pg_alg(args)
    else:
        play(args)
