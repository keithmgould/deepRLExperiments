import gym
import pdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Policy():
    def __init__(self, hparams, session):
        self.session = session
        with tf.name_scope('policy'):
            self.actions = tf.placeholder(tf.int32, name="actions")
            self.advantages = tf.placeholder(tf.float32, name="advantages")
            self.observation = tf.placeholder(tf.float32, shape=[None, hparams['observation_size']])
            self.build_network(hparams)
            self.sample = tf.reshape(tf.multinomial(self.network, 1), [])
            self.build_trainer(hparams)

    def build_network(self, hparams):
        hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.observation,
                num_outputs=hparams['hidden_size1'],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer())

        self.network = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=hparams['num_actions'],
                activation_fn=None)

    def build_trainer(self, hparams):
        loss = self.build_loss()
        optimizer = tf.train.RMSPropOptimizer(hparams['policy_learning_rate'])
        self.train = optimizer.minimize(loss)

    def build_loss(self):
        action_prob = self.build_action_prob()
        reduced_sum = tf.reduce_sum(tf.multiply(action_prob, self.advantages))
        return -reduced_sum

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
    def build_action_prob(self):
        with tf.name_scope('action_prob'):
            softmax = tf.nn.softmax(self.network) # => [[.75, .25], [.5, .5], [.8, .2],...]
            log_prob = tf.log(softmax) # => [[-1.8 -0.1],[-0.1 -2.],[-2.1 -0.1],...]
            shape = tf.shape(log_prob) # => [totalStepsInBatchCount, actionSpaceCount]
            ep_range = tf.range(0, shape[0]) # => [0, 1, 2, ..., totalStepsInBatchCount]
            flattened = tf.reshape(log_prob, [-1]) # flatten, putting ALL actions (actions we took AND DID NOT TAKE) side by side
            indices = ep_range * shape[1] + self.actions # given flattened, picks the indexes of the actions taken
            action_prob = tf.gather(flattened, indices) # just choose the actions we actually took (remove actions we did not take)
            return action_prob

    # given an observation, pick an action
    def observe_and_act(self, observation):
        return self.session.run(self.sample, feed_dict={self.observation: [observation]})

    def train_net(self, batch_observations, batch_actions, batch_advantages):
        feed = {
            self.observation: batch_observations,
            self.actions: batch_actions,
            self.advantages: batch_advantages
        }

        self.session.run(self.train, feed_dict=feed)

class ValueEstimator():
    def __init__(self, hparams, session):
        with tf.name_scope('policy_estimator'):
            self.session = session
            self.advantages = tf.placeholder(tf.float32, name="advantages")
            self.observation = tf.placeholder(tf.float32, shape=[None, hparams['observation_size']])
            self.build_network(hparams)
            self.build_trainer(hparams)

    def build_network(self, hparams):
        hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.observation,
                num_outputs=hparams['hidden_size1'],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer())

        self.estimation = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=1,
                activation_fn=None)

    def build_trainer(self, hparams):
        loss = self.build_loss()
        optimizer = tf.train.RMSPropOptimizer(hparams['value_learning_rate'])
        self.train = optimizer.minimize(loss)

    def build_loss(self):
        reduced_sum = tf.reduce_sum(tf.multiply(self.estimation, self.advantages))
        return -reduced_sum

    def estimate(self, batch_observations):
        feed = { self.observation: batch_observations }
        return self.session.run(self.estimation, feed_dict = feed)[:,0]

    def train_net(self, batch_observations, batch_advantages):
        feed = {
            self.observation: batch_observations,
            self.advantages: batch_advantages
        }
        return self.session.run(self.train, feed_dict = feed)

def policy_rollout(env, policy):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    observations, actions, rewards  = [], [], []

    while not done:
        # env.render()
        observations.append(observation)

        action = policy.observe_and_act(observation)
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
            'hidden_size1': 36,
            'num_actions': env.action_space.n,
            'policy_learning_rate': 0.01,
            'value_learning_rate': 0.01,
    }

    # environment params
    eparams = {
            'num_batches': 100,
            'ep_per_batch': 10
    }

    with tf.Graph().as_default(), tf.Session() as sess:
        policy = Policy(hparams, sess)
        valueEstimator = ValueEstimator(hparams, sess)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs/", graph=tf.get_default_graph())

        episode_lengths = []

        for batch in range(eparams['num_batches']):
            print('\n=====\nBATCH {}\n===='.format(batch))
            batch_observations, batch_actions, batch_rewards, step_counts = [], [], [], []

            for ep_index in range(eparams['ep_per_batch']):
                observations, actions, rewards = policy_rollout(env, policy)
                step_count = len(observations)
                episode_lengths.append(step_count)
                step_counts.append(step_count)
                print('Episode {} steps: {}'.format((ep_index+1)+(10 * batch),step_count))
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                advantages = [len(rewards)] * len(rewards)
                batch_rewards.extend(advantages)

            print("Avg Step count: {}".format(np.average(step_counts)))
            batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-10)
            batch_estimates = valueEstimator.estimate(batch_observations)
            batch_advantages = batch_rewards - batch_estimates
            valueEstimator.train_net(batch_observations, batch_advantages)
            policy.train_net(batch_observations, batch_actions, batch_advantages)
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(episode_lengths, 'r.', linewidth=1, label="Data")
        ax.plot(calc_moving_average(episode_lengths), 'b.', linewidth=1, label="Running Avg")
        ax.legend()
        fig.savefig('reinforce_baseline_batch_ep_lengths.png')
        plt.close(fig)

if __name__ == "__main__":
    main()
