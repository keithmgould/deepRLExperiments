import pdb
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    #    action_prob = [lpa,lpc,lpf]
    #    return action_prob
    def build_action_prob(self, logits, actions):
        with tf.name_scope('action_log_prob'):
            softmax = tf.nn.softmax(logits) # => [[.75, .25], [.5, .5], [.8, .2],...]
            log_prob = tf.log(softmax) # => [[-1.8 -0.1],[-0.1 -2.],[-2.1 -0.1],...]
            shape = tf.shape(log_prob) # => [totalStepsInBatchCount, actionSpaceCount]
            ep_range = tf.range(0, shape[0]) # => [0, 1, 2, ..., totalStepsInBatchCount]
            flattened = tf.reshape(log_prob, [-1]) # flatten, putting ALL actions (actions we took AND DID NOT TAKE) side by side
            indices = ep_range * shape[1] + actions # given flattened, picks the indexes of the actions taken
            action_prob = tf.gather(flattened, indices) # just choose the actions we actually took (remove actions we did not take)
        return action_prob

    def build_graph(self, hparams):
        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self._observations,
            num_outputs=hparams['hidden_size1'],
            activation_fn=tf.nn.relu,
            weights_initializer=_initializer)

        with tf.variable_scope('final_layer'):
            logits = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=hparams['num_actions'],
                activation_fn=None)

        return logits

    def __init__(self, hparams, sess):

        # initialization
        self._s = sess

        # placeholders
        self._observations = tf.placeholder(tf.float32, shape=[None, hparams['observation_size']], name="observations")
        self._actions = tf.placeholder(tf.int32, name="actions")
        self._advantages = tf.placeholder(tf.float32, name="advantages")

        # build graph, returning pointer to final logit layer
        logits = self.build_graph(hparams)
        self._logits = logits

        # op to sample an action
        self._sample = tf.reshape(tf.multinomial(logits, 1), [], name="sample")

        # self.tv = tf.trainable_variables('final_layer')
        # self.my_weights = tf.Variable(self.tv[0])
        # self.my_biases = tf.Variable(self.tv[1])


        # build logic for determining action probabilities
        # returning pointer to action probabilities
        with tf.name_scope('loss'):
            action_prob = self.build_action_prob(logits, self._actions)
            loss = -tf.reduce_sum(tf.multiply(action_prob, self._advantages))
            optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
            # self.grads_and_vars = optimizer.compute_gradients(loss)

            # self._train = optimizer.apply_gradients(vars_and_grads)
            self._train = optimizer.minimize(loss)

    # get one action, by sampling
    def act(self, observation):
        return self._s.run(self._sample, feed_dict={self._observations: [observation]})

    def train_step(self, batch_observations, batch_actions, batch_advantages):
        batch_feed = {
            self._observations: batch_observations,
            self._actions: batch_actions,
            self._advantages: batch_advantages
        }

        pdb.set_trace()

        return self._s.run(self._train, feed_dict=batch_feed)


def policy_rollout(env, agent):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    observations, actions, rewards = [], [], []

    while not done:

        # env.render()
        observations.append(observation)

        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)

    return observations, actions, rewards

def calc_moving_average(mylist):
    N = 10
    cumsum, moving_avgs = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_avgs.append(moving_ave)
    return moving_avgs

def main():
    env = gym.make('CartPole-v0')

    # hyper parameters
    hparams = {
            'observation_size': env.observation_space.shape[0],
            'hidden_size1': 200,
            'num_actions': env.action_space.n,
            'learning_rate': 0.01
    }

    # environment params
    eparams = {
            'num_batches': 1000,
            'ep_per_batch': 10
    }

    with tf.Graph().as_default(), tf.Session() as sess:
        agent = PolicyGradientAgent(hparams, sess)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs/", graph=tf.get_default_graph())
        episode_lengths = []
        for batch in range(eparams['num_batches']):
            print('=====\nBATCH {}\n===='.format(batch))
            batch_observations, batch_actions, batch_rewards = [], [], []
            for ep_index in range(eparams['ep_per_batch']):
                observations, actions, rewards = policy_rollout(env, agent)
                episode_length = len(observations)
                episode_lengths.append(episode_length)
                print('Episode {} steps: {}'.format((ep_index+1)+(10 * batch), episode_length))
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                advantages = [len(rewards)] * len(rewards)
                batch_rewards.extend(advantages)

            batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-10)
            train_results = agent.train_step(batch_observations, batch_actions, batch_rewards)
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(episode_lengths, 'r.', linewidth=1, label="Data")
        ax.plot(calc_moving_average(episode_lengths), 'b.', linewidth=1, label="Running Avg")
        ax.legend()
        fig.savefig('reinforce_batch__no_normalize_ep_lengths.png')
        plt.close(fig)


if __name__ == "__main__":
    main()
