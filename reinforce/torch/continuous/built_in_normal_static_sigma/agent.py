import sys
import math
import pdb
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable, Function
from policy import Policy
from torchviz import make_dot



pi = Variable(torch.FloatTensor([math.pi]))

class Agent:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def create_distribution(self, observation):
        observation = Variable(observation)
        mu = self.model(observation)
        sigma = 1
        self.episode_sigmas.append(sigma)
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
        loss = 0

        # for i in range(len(rewards)):
        for action, observation, r in zip(actions, observations, returns):
            dist = self.create_distribution(observation)
            loss += dist.log_prob(action) * r
        loss = loss / len(returns)
        loss = -loss

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return loss
