import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow.contrib.keras as tfck

EPISODES = 1000

class ReinforcePolicyNet():

    def __init__(self, state_shape=[4], action_size=2, fc_layer_sizes=[24, 24], name_scope='policy_net'):
        self.name_scope = name_scope
        with tf.variable_scope(self.name_scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=(None, *state_shape))
            self.advantage = tf.placeholder(dtype=tf.float32, shape=(None, action_size))
    
            self.layer = tf.contrib.layers.flatten(self.state)
            for layer_size in fc_layer_sizes:
                self.layer = tf.contrib.layers.fully_connected(self.layer, layer_size, activation_fn=tf.nn.relu)
            self.logits = tf.contrib.layers.fully_connected(self.layer, action_size, activation_fn=None)
            self.policy = tf.nn.softmax(self.logits)

            # self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.advantage), axis=0)
            # self.loss = tf.contrib.keras.metrics.categorical_crossentropy(y_true=self.advantage, y_pred=self.policy)
            self.loss = tf.reduce_mean(tf.multiply(-tf.log(self.policy), self.advantage))
            self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.loss)
            

# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.use_original = False # True, False
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 24, 24

        # create model for policy network
        if self.use_original:
            self.model = self.build_model_tf_contrib_keras() # self.build_model() - self.build_model_tf_contrib_keras()
        else:
            self.model = ReinforcePolicyNet()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        # if self.load_model:
        #     self.model.load_weights("./save_model/cartpole_reinforce.h5")

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set 
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a). 
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    def build_model_tf_contrib_keras(self):
        model = tfck.models.Sequential()
        model.add(tfck.layers.Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(tfck.layers.Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(tfck.layers.Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        # model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set 
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a). 
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer='adam')
        return model        


    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        if self.use_original:
            policy = self.model.predict(state, batch_size=1).flatten()
        else:
            policy = self.model.policy.eval(feed_dict={self.model.state: state})[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        if self.use_original:
            self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        else:
            _ = tf.get_default_session().run([self.model.train_step], feed_dict={self.model.state:update_inputs, self.model.advantage:advantages})
            # for ui, a in zip(update_inputs, advantages):
            #     _ = tf.get_default_session().run([self.model.train_step], feed_dict={self.model.state:[ui], self.model.advantage:[a]})
        self.states, self.actions, self.rewards = [], [], []

if __name__ == "__main__":
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)

            score += reward
            state = next_state

            if done:
                # every episode, agent learns from sample returns
                agent.train_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/cartpole_reinforce.png")
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        # if e % 50 == 0:
        #     agent.model.save_weights("./save_model/cartpole_reinforce.h5")
