import gym
import math
import numpy as np
import tensorflow as tf
import pdb

from tensorflow.python.ops import random_ops

def _initializer(shape, dtype=tf.float32, partition_info=None):
     return random_ops.random_normal(shape)

class PolicyGradientAgent(object):

    #######################################
    # Get log probs of actions from episode.
    # Lots of lines because its done inside
    # of TensorFlow (not Python)
    # given:
    #    logits = [[a,b],[c,d],[e,f]]
    #    actions = [0,0,1]
    # produce:
    #    lpa = log prob of 'a'
    #    return [lpa,lpc,lpf]
    def build_log_probs(self, logits, actions):
        softmax = tf.nn.softmax(logits) # => [[.75, .25, ...]]
        log_prob = tf.log(softmax) # => [[-1.8 -0.1, ...]]
        shape = tf.shape(log_prob) # => [totalStepsInEpisode, actionSpaceCount]
        ep_range = tf.range(0, shape[0]) # => [0, 1, 2, ..., totalStepsInEpisode]
        flattened = tf.reshape(log_prob, [-1]) # flatten, putting ALL actions (actions we took AND DID NOT TAKE) side by side
        indices = ep_range * shape[1] + actions # given flattened, picks the indexes of the actions taken
        action_prob = tf.gather(flattened, indices) # just choose the actions we actually took (remove actions we did not take)
        return action_prob

    def build_graph(self, hparams):
        hidden = tf.layers.dense(
            self.observations,
            hparams['hidden_size1'],
            activation=tf.nn.relu,
            name="hidden")
        return tf.layers.dense(
            hidden,
            hparams['num_actions'],
            name="logits")

    def __init__(self, hparams, sess):
        # optimizer
        optimizer = tf.train.AdamOptimizer(hparams['learning_rate'])

        # initialization
        self.s = sess

        # placeholders
        self.observations = tf.placeholder(tf.float32, shape=[None, hparams['observation_size']], name="Observations")
        self.actions = tf.placeholder(tf.int32, name="Actions")
        self.advantages = tf.placeholder(tf.float32, name="Advantages")

        # build graph, returning pointer to final logit layer
        logits = self.build_graph(hparams)

        # op to sample an action
        with tf.name_scope('sample'):
            self.sample = tf.reshape(tf.multinomial(logits, 1), []), logits

        # build logic for determining action probabilities
        # returning pointer to action probabilities
        with tf.name_scope('log_probs'):
            log_probs = self.build_log_probs(logits, self.actions)

        with tf.name_scope('loss'):
            loss = -tf.reduce_sum(tf.multiply(log_probs, self.advantages))

        self.train = optimizer.minimize(loss), loss

    # get one action, by sampling
    def act(self, observation):
        return self.s.run(self.sample, feed_dict={self.observations: [observation]})

    def train_step(self, observations, actions, advantages):
        batch_feed = {
            self.observations: observations,
            self.actions: actions,
            self.advantages: advantages
        }

        return self.s.run(self.train, feed_dict=batch_feed)


def policy_rollout(env, agent):
    observation, reward, done = env.reset(), 0, False
    observations, actions, rewards= [], [], []

    while not done:
        observations.append(observation)
        action, logit = agent.act(observation)
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
            foo, loss = policy.train_step(observations, actions, advantages)
            print('Episode {} steps: {}, loss: {}'.format(ep_index,len(observations), loss))

if __name__ == "__main__":
    main()
