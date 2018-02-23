# modified from:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import pdb
import argparse
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--resume', action='store_true',
                    help='resume from saved network (default: False)')
args = parser.parse_args()

env = gym.make('CartPole-v1')
env.seed(args.seed)
torch.manual_seed(args.seed)

PATH = "./saved_model"

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.observations = []
        self.actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


policy = Policy()

if args.resume:
    policy.load_state_dict(torch.load(PATH))

optimizer = optim.Adam(policy.parameters(), lr=1e-3)

def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(Variable(state))
    m = torch.distributions.Categorical(probs)
    return m.sample()

def normalize_rewards(raw_rewards):
  R = 0
  rewards = []
  for r in policy.rewards[::-1]:
      # apply the discount
      R = r + args.gamma * R
      rewards.insert(0, R)

  # give rewards a zero mean, and a std of 1
  # add a very (very) small number to prevent division by zero.
  rewards = np.array(rewards)
  rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

  return rewards

def finish_episode():
    returns = normalize_rewards(policy.rewards)

    loss = 0
    for action, observation, r in zip(policy.actions, policy.observations, returns):
        var = Variable(torch.from_numpy(observation).float())
        probs = policy(var)
        m = torch.distributions.Categorical(probs)
        loss += m.log_prob(action) * r
    loss = loss / len(returns)
    loss = -loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.actions[:]
    del policy.observations[:]
    torch.save(policy.state_dict(), PATH)

def print_status(i_episode):
    print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(i_episode, t, running_reward))

running_reward = 10
for i_episode in count(100):
    state = env.reset()
    episode_reward_sum = 0

    for t in range(1000):
        action = select_action(state)
        state, reward, done, _ = env.step(action.data[0])
        episode_reward_sum += reward
        policy.rewards.append(reward)
        policy.actions.append(action)
        policy.observations.append(state)

        if done:
            break

    running_reward = running_reward * 0.99 + episode_reward_sum * 0.01
    finish_episode()
    print_status(i_episode)
