import gym
import math
import numpy as np
import tensorflow as tf
import shutil
import pdb

from tensorflow.python.ops import random_ops

def _initializer(shape, dtype=tf.float32, partition_info=None):
     return random_ops.random_normal(shape)

class PolicyGradientAgent(object):

    def build_graph(self, hparams):
        hidden = tf.layers.dense(
            self.observations,
            hparams['hidden_size1'],
            activation=tf.nn.relu,
            name='hidden')
        weights = tf.get_default_graph().get_tensor_by_name('hidden/kernel:0')
        bias = tf.get_default_graph().get_tensor_by_name('hidden/bias:0')
        tf.summary.histogram("weights", weights, family="hidden")
        tf.summary.histogram("bias", bias, family="hidden")
        tf.summary.histogram("activations", hidden, family="hidden")

        logits = tf.layers.dense(
            hidden,
            hparams['num_actions'],
            name="logits")

        weights = tf.get_default_graph().get_tensor_by_name('logits/kernel:0')
        bias = tf.get_default_graph().get_tensor_by_name('logits/bias:0')
        tf.summary.histogram("weights", weights, family="logits")
        tf.summary.histogram("bias", bias, family="logits")
        tf.summary.histogram("activations", logits, family="logits")


        return logits

    def __init__(self, hparams, sess):
        # optimizer
        optimizer = tf.train.AdamOptimizer(hparams['learning_rate'])

        # initialization
        self.s = sess

        # placeholders
        self.observations = tf.placeholder(tf.float32, shape=[None, hparams['observation_size']], name="Observations")
        self.actions = tf.placeholder(tf.int32, name="Actions")
        self.advantages = tf.placeholder(tf.float32, name="Advantages")

        logits = self.build_graph(hparams)
        self.logits = logits # tmp

        dist = tf.distributions.Categorical(logits=logits)

        with tf.name_scope('sample'):
            self.sample = tf.reshape(dist.sample(),[])

        with tf.name_scope('loss'):
            log_probs = dist.log_prob(tf.reshape(self.actions,[-1]))
            loss = -tf.reduce_sum(tf.multiply(log_probs, tf.stop_gradient(self.advantages)))


        self.grads_and_vars = optimizer.compute_gradients(loss)
        hidden_weight_grads = self.grads_and_vars[0][0]
        hidden_bias_grads = self.grads_and_vars[1][0]
        logit_weight_grads = self.grads_and_vars[2][0]
        logit_bias_grads = self.grads_and_vars[3][0]
        tf.summary.histogram("weightGrads", hidden_weight_grads, family="hidden")
        tf.summary.histogram("biasGrads", hidden_bias_grads, family="hidden")
        tf.summary.histogram("weightGrads", logit_weight_grads, family="logits")
        tf.summary.histogram("biasGrads", logit_weight_grads, family="logits")
        self.train = optimizer.apply_gradients(self.grads_and_vars)
        # self.train = optimizer.minimize(loss)

        self.summaries = tf.summary.merge_all()

    # get one action, by sampling
    def act(self, observation):
        return self.s.run(self.sample, feed_dict={self.observations: [observation]})

    def train_step(self, observations, actions, advantages):
        batch_feed = {
            self.observations: observations,
            self.actions: actions,
            self.advantages: advantages
        }

        # pdb.set_trace()

        return self.s.run([self.train, self.summaries], feed_dict=batch_feed)

def policy_rollout(env, agent):
    observation, reward, done = env.reset(), 0, False
    observations, actions, rewards= [], [], []

    while not done:
        observations.append(observation)
        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)

    return observations, actions, rewards

def calculate_advantages(rewards):
    advantages = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
      running_add = running_add * 0.99 + rewards[t]
      advantages[t] = running_add
    return advantages

def main():

    # delete the logs directory so no dups for tensorboard.
    shutil.rmtree("./logs", True)

    env = gym.make('CartPole-v0')
    max_episodes = 10000

    hparams = {
            'observation_size': env.observation_space.shape[0],
            'hidden_size1': 32,
            'num_actions': env.action_space.n,
            'learning_rate': 0.01
    }

    with tf.Graph().as_default(), tf.Session() as sess:
        policy = PolicyGradientAgent(hparams, sess)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs/", graph=tf.get_default_graph())

        for ep_index in range(max_episodes):
            observations, actions, rewards = policy_rollout(env, policy)
            advantages = calculate_advantages(rewards)
            foo, summaries = policy.train_step(observations, actions, advantages)
            writer.add_summary(summaries, global_step=ep_index)
            print('Episode {} steps: {}'.format(ep_index,len(observations)))

if __name__ == "__main__":
    main()
