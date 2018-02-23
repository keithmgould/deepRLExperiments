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
parser.add_argument('--seed', type=int, default=123, metavar='N',
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

    reward_sum = np.sum(rewards)
    reward_sums.append(reward_sum)
    print("Episode: {}, reward: {}, avg: {}".format(
        i_episode,
        reward_sum,
        np.average(reward_sums[-10:])
    ))

env.close()
