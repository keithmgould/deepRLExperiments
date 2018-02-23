import argparse, math, os
import numpy as np
import gym
import roboschool
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import pdb
from agent import Agent

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env_name', type=str, default='RoboschoolInvertedPendulum-v2')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 2000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of hidden neurons (default: 128)')


args = parser.parse_args()

env_name = args.env_name
env = gym.make(env_name)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = Agent(args.hidden_size, env.observation_space.shape[0], env.action_space)
reward_sums = []
for i_episode in range(args.num_episodes):
    agent.episode_mus = []
    agent.episode_sigmas = []
    observation = torch.Tensor([env.reset()])
    observations = []
    actions = []
    rewards = []
    for t in range(args.num_steps):
        action = agent.select_action(observation)
        next_observation, reward, done, _ = env.step(action)
        observations.append(observation)
        rewards.append(reward)
        actions.append(action[0])
        observation = torch.Tensor([next_observation])

        if done:
            break

    loss = agent.update_parameters(observations, actions, rewards, args.gamma)

    if i_episode % 750 == 0 and i_episode > 0:
        sigh = np.array(agent.l1_grad_mean_max_min)
        l1_means = sigh[:,0]
        l1_maxes = sigh[:,1]
        l1_mins = sigh[:,2]

        sigh2 = np.array(agent.l1_grad_mean_max_min)
        l2_means = sigh2[:,0]
        l2_maxes = sigh2[:,1]
        l2_mins = sigh2[:,2]
        f, axarr = plt.subplots(10, sharex=True, figsize=(10,10))
        axarr[0].plot(l1_means)
        axarr[0].grid(True)
        axarr[0].set_title('l1 grad means')
        axarr[1].plot(l1_maxes)
        axarr[1].grid(True)
        axarr[1].set_title('l1 grad maxes')
        axarr[2].plot(l1_mins)
        axarr[2].grid(True)
        axarr[2].set_title('l1 grad mins')
        axarr[3].plot(l2_mins)
        axarr[3].grid(True)
        axarr[3].set_title('l2 grad means')
        axarr[4].plot(l2_maxes)
        axarr[4].grid(True)
        axarr[4].set_title('l2 grad maxes')
        axarr[5].plot(l2_mins)
        axarr[5].grid(True)
        axarr[5].set_title('l2 grad mins')
        axarr[6].plot(agent.losses)
        axarr[6].grid(True)
        axarr[6].set_title('Loss')
        axarr[7].plot(agent.ep_len)
        axarr[7].grid(True)
        axarr[7].set_title('Episide Length')
        axarr[8].plot(agent.sigmas)
        axarr[8].grid(True)
        axarr[8].set_title('Sigmas')
        axarr[9].plot(agent.mus)
        axarr[9].grid(True)
        axarr[9].set_title('Mus')
        plt.subplots_adjust(hspace = 1)
        plt.xlabel("Episode #")
        plt.savefig("./plots.png")

    reward_sum = np.sum(rewards)
    reward_sums.append(reward_sum)
    print("Episode: {}, reward: {}, avg: {}".format(
        i_episode,
        reward_sum,
        np.average(reward_sums[-10:])
    ))

env.close()
