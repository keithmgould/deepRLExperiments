import numpy as np
import gym
import pdb, shutil
import roboschool
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import random_ops

def _initializer(shape, dtype=tf.float32, partition_info=None):
     return random_ops.random_normal(shape)

class Policy:
  def __init__(self, session, observation_size, lowest_action, highest_action):
    self.session = session
    self.lowest_action = lowest_action
    self.highest_action = highest_action
    optimizer = tf.train.AdamOptimizer(.001)
    self.observations = tf.placeholder(tf.float32, shape=[None, observation_size], name="observations")
    self.actions = tf.placeholder(tf.float32, name="actions")
    self.returns = tf.placeholder(tf.float32, name="returns")
    normal = self.build_graph()

    with tf.name_scope('sampler'):
        self.action = tf.reshape(normal.sample(),[])

    with tf.name_scope('log_probs'):
        self.log_probs = normal.log_prob(self.actions)

    with tf.name_scope('loss'):
        loss = -tf.reduce_mean(tf.multiply(self.log_probs, self.returns))

    self.loss = loss
    gvs = optimizer.compute_gradients(loss)

    hidden_weight_grads = gvs[0][0]
    hidden_bias_grads = gvs[1][0]
    mu_weight_grads = gvs[2][0]
    mu_bias_grads = gvs[3][0]
    sigma_weight_grads = gvs[4][0]
    sigma_bias_grads = gvs[5][0]
    tf.summary.histogram("weightGrads", hidden_weight_grads, family="hidden")
    tf.summary.histogram("biasGrads", hidden_bias_grads, family="hidden")
    tf.summary.histogram("weightGrads", mu_weight_grads, family="mu")
    tf.summary.histogram("biasGrads", mu_weight_grads, family="mu")
    tf.summary.histogram("weightGrads", sigma_weight_grads, family="sigma")
    tf.summary.histogram("biasGrads", sigma_weight_grads, family="sigma")

    self.trainMe = optimizer.apply_gradients(gvs)
    self.summaries = tf.summary.merge_all()

  def tensorboard_histogram_store(self, family, layer):
    weights = tf.get_default_graph().get_tensor_by_name(family + '/kernel:0')
    bias = tf.get_default_graph().get_tensor_by_name(family + '/bias:0')
    tf.summary.histogram("weights", weights, family=family)
    tf.summary.histogram("bias", bias, family=family)
    tf.summary.histogram("activations", layer, family=family)

  def build_graph(self):
    hidden = tf.layers.dense(self.observations, 128, tf.nn.relu, name="hidden")
    self.tensorboard_histogram_store("hidden", hidden)
    mu = tf.layers.dense(hidden,1, tf.nn.tanh, name="mu")
    self.tensorboard_histogram_store("mu", mu)
    sigma_sq = tf.layers.dense(hidden,1, tf.nn.softplus, name="sigma")
    self.tensorboard_histogram_store("sigma", sigma_sq)

    sigma = tf.sqrt(sigma_sq)
    self.flat_sigma = tf.reshape(sigma,[-1])
    self.flat_mu = tf.reshape(mu,[-1])
    return tf.distributions.Normal(self.flat_mu, self.flat_sigma)

  def select_action(self, observation):
    feed = { self.observations: [observation] }
    action = self.session.run(self.action, feed_dict=feed)
    return np.clip(action, self.lowest_action, self.highest_action)

  def update_parameters(self, observations, actions, returns, ep_index):
    feed = {
      self.observations: observations,
      self.actions: actions,
      self.returns: returns,
    }

    log_probs, _, summaries, loss, mus, sigmas = self.session.run([
      self.log_probs,
      self.trainMe,
      self.summaries,
      self.loss,
      self.flat_mu,
      self.flat_sigma], feed_dict = feed)

    return summaries, loss

# End of Policy
#----------------------------------------------------------------
# Beginning of Agent

class Agent:
  def __init__(self):
    self.env = gym.make('RoboschoolInvertedPendulum-v1')
  def print_episode_results(self, ep_index, action_lengths, loss):
    print("Episode {0}. Steps {1}. Avg {2:.2f}. Loss {3:.2f}".format(
      ep_index,
      action_lengths[-1],
      np.average(action_lengths[-10:]),
      loss
    ))

  def run(self):
    with tf.Graph().as_default(), tf.Session() as session:
      policy = Policy(session, self.env.observation_space.shape[0], self.env.action_space.low[0], self.env.action_space.high[0])
      session.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter("./logs/", graph=tf.get_default_graph())
      action_lengths = []
      for ep_index in range(5000):
        observations, actions, rewards = self.policy_rollout(policy)
        returns = self.discount_rewards(rewards)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        summaries, loss = policy.update_parameters(observations, actions, returns, ep_index)
        writer.add_summary(summaries, global_step=ep_index)
        action_lengths.append(len(actions))
        self.print_episode_results(ep_index, action_lengths,loss)

  def policy_rollout(self, policy):
    observation, reward, done = self.env.reset(), 0, False
    observations, actions, rewards  = [], [], []

    while not done:
      action = policy.select_action(observation)
      wrapped_action = np.array([action])
      observation, reward, done, _ = self.env.step(wrapped_action)
      observations.append(observation)
      actions.append(action)
      rewards.append(reward)

    return observations, actions, rewards

  def discount_rewards(self, rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
      running_add = running_add * 0.99 + rewards[t]
      discounted_rewards[t] = running_add
    return discounted_rewards

# End of Agent
#----------------------------------------------------------------
# Beginning of Main

shutil.rmtree("./logs", True)
agent = Agent()
agent.run()
