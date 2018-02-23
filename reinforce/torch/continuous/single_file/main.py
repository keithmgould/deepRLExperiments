import argparse, math, os, sys
import numpy as np
import pdb
import gym
import roboschool
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

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

class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = F.relu(x)
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)
        sigma_sq = F.softplus(sigma_sq)

        return mu, sigma_sq


# End Policy
#----------------------------------------------
# Begin Agent

pi = Variable(torch.FloatTensor([math.pi]))

class Agent:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        self.l1_nz = []
        self.l2_nz = []
        self.l1_grad_mean_max_min = []
        self.l2_grad_mean_max_min = []
        self.losses = []
        self.ep_len = []
        self.sigmas = []
        self.mus = []
        self.episode_sigmas = []
        self.episode_mus = []

    def create_distribution(self, observation):
        observation = Variable(observation)
        mu, sigma_sq = self.model(observation)
        sigma = sigma_sq.sqrt()
        self.episode_sigmas.append(sigma.data[0][0])
        self.episode_mus.append(mu.data[0][0])
        return torch.distributions.Normal(mu, sigma)

    def select_action(self, observation):
        dist = self.create_distribution(observation)
        return dist.sample().data[0]

    def discount_rewards(self, rewards, gamma):
        stepReturn = 0
        stepReturns = []
        for i in range(len(rewards)):
            stepReturn = gamma * stepReturn + rewards[i]
            stepReturns.append(stepReturn)
        return list(reversed(stepReturns))

    def update_parameters(self, observations, actions, rewards, gamma):
        returns = self.discount_rewards(rewards, gamma)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10)
        loss = 0

        # for i in range(len(rewards)):
        for action, observation, r in zip(actions, observations, returns):
            dist = self.create_distribution(observation)
            loss += dist.log_prob(action) * r
        loss = loss / len(returns)
        loss = -loss
        self.losses.append(loss.data[0][0])
        self.ep_len.append(len(returns))
        self.mus.append(np.mean(self.episode_mus))
        self.sigmas.append(np.mean(self.episode_sigmas))
        self.episode_sigmas = []
        self.episode_mus = []
        self.optimizer.zero_grad()
        loss.backward()
        for f in self.model.linear1.parameters():
            if(np.size(f.data.numpy()) == 512):
                foo = f.grad.data.numpy()
                bar = np.count_nonzero(foo)
                self.l1_nz.append(bar)
                mmm = [np.mean(foo), np.max(foo), np.min(foo)]
                self.l1_grad_mean_max_min.append(mmm)
        for f in self.model.linear2_.parameters():
            if(np.size(f.data.numpy()) == 128):
                foo = f.grad.data.numpy()
                bar = np.count_nonzero(foo)
                self.l2_nz.append(bar)
                mmm = [np.mean(foo), np.max(foo), np.min(foo)]
                self.l2_grad_mean_max_min.append(mmm)

        self.optimizer.step()
        return loss

# End Agent
#-----------------------------------
# Begin Main

agent = Agent(args.hidden_size, env.observation_space.shape[0], env.action_space)
reward_sums = []
for i_episode in range(args.num_episodes):
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
