import numpy as np
import gym
import pdb, shutil
import roboschool
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow.python.ops import random_ops

class Policy:
  def __init__(self, session, observation_size, lowest_action, highest_action):
    self.session = session
    self.lowest_action = lowest_action
    self.highest_action = highest_action
    optimizer = tf.train.AdamOptimizer(.001)

    # placeholders
    self.observations = tf.placeholder(tf.float32, shape=[None, observation_size], name="observations")
    self.actions = tf.placeholder(tf.float32, name="actions")
    self.returns = tf.placeholder(tf.float32, name="returns")

    # network generates mu and sigma
    hidden = tf.layers.dense(self.observations, 128, tf.nn.relu, name="hidden")
    mu = tf.layers.dense(hidden,1, tf.nn.tanh, name="mu")
    self.mu = tf.reshape(mu,[-1])
    sigma_theta = tf.get_variable("sigma_theta",[32], initializer=tf.zeros_initializer())
    sigma = tf.reduce_sum(sigma_theta)
    self.sigma = tf.exp(sigma)

    # normal distribution utilizes mu and sigma
    normal = tf.distributions.Normal(self.mu, self.sigma)

    with tf.name_scope('sampler'):
        self.action = tf.reshape(normal.sample(),[])

    with tf.name_scope('log_probs'):
        self.log_probs = normal.log_prob(self.actions)

    with tf.name_scope('loss'):
        loss = -tf.reduce_mean(tf.multiply(self.log_probs, self.returns))

    self.trainMe = optimizer.minimize(loss)

  def select_action(self, observation):
    feed = { self.observations: [observation] }
    mu, sigma = self.session.run([self.mu, self.sigma], feed_dict=feed)
    mu= mu[0]

    action = np.random.normal(mu, sigma)
    return np.clip(action, self.lowest_action, self.highest_action)

  def update_parameters(self, observations, actions, returns):
    feed = {
      self.observations: observations,
      self.actions: actions,
      self.returns: returns,
    }

    self.session.run(self.trainMe, feed_dict = feed)

# End of Policy
#----------------------------------------------------------------
# Beginning of Agent

class Agent:
  def __init__(self):
    self.env = gym.make('RoboschoolInvertedPendulum-v1')

  def run(self):
    with tf.Graph().as_default(), tf.Session() as session:
      policy = Policy(session, self.env.observation_space.shape[0], self.env.action_space.low[0], self.env.action_space.high[0])
      session.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter("./logs/", graph=tf.get_default_graph())
      for batch in range(1000):
        print('=====\nBATCH {}\n===='.format(batch))
        batch_observations, batch_actions, batch_rewards = [], [], []
        ep_lengths = []
        for ep_index in range(10):
          observations, actions, rewards = self.policy_rollout(policy)
          batch_observations.extend(observations)
          batch_actions.extend(actions)
          advantages = [len(rewards)] * len(rewards)
          batch_rewards.extend(advantages)
          ep_length = len(actions)
          ep_lengths.append(ep_length)
          print('Episode {} steps: {}'.format((ep_index+1)+(10 * batch), ep_length))
        batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-10)
        policy.update_parameters(batch_observations, batch_actions, batch_rewards)
        print("AVG: {0:.2f}".format(np.mean(np.array(ep_lengths))))

  def policy_rollout(self, policy):
    observation, reward, done = self.env.reset(), 0, False
    observations, actions, rewards  = [], [], []

    while not done:
      action = policy.select_action(observation)
      wrapped_action = np.array([action])
      observations.append(observation)
      actions.append(action)
      observation, reward, done, _ = self.env.step(wrapped_action)
      rewards.append(reward)

    return observations, actions, rewards

# End of Agent
#----------------------------------------------------------------
# Beginning of Main

shutil.rmtree("./logs", True)
agent = Agent()
agent.run()
