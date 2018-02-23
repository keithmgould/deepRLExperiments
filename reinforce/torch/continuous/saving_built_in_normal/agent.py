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
from tempfile import TemporaryFile



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
        self.intermediates = []

    def create_distribution(self, observation):
        observation = Variable(observation)
        mu, sigma_sq, intermediate = self.model(observation)
        self.intermediates.append(intermediate)
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
        loss = 0

        # for i in range(len(rewards)):
        log_probs = []
        for action, observation, r in zip(actions, observations, returns):
            dist = self.create_distribution(observation)
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            loss += log_prob * r
        loss = loss / len(returns)
        loss = -loss
        self.losses.append(loss.data[0][0])
        self.ep_len.append(len(returns))
        self.mus.append(np.mean(self.episode_mus))
        self.sigmas.append(np.mean(self.episode_sigmas))

        self.optimizer.zero_grad()
        loss.backward()
        saved_episode = []
        for f in self.model.linear1.parameters():
            saved_episode.append(f.data.numpy())
            if(np.size(f.data.numpy()) == 512):
                foo = f.grad.data.numpy()
                bar = np.count_nonzero(foo)
                self.l1_nz.append(bar)
                mmm = [np.mean(foo), np.max(foo), np.min(foo)]
                self.l1_grad_mean_max_min.append(mmm)
        for f in self.model.linear2.parameters():
            saved_episode.append(f.data.numpy())
        for f in self.model.linear2_.parameters():
            saved_episode.append(f.data.numpy())
            if(np.size(f.data.numpy()) == 128):
                foo = f.grad.data.numpy()
                bar = np.count_nonzero(foo)
                self.l2_nz.append(bar)
                mmm = [np.mean(foo), np.max(foo), np.min(foo)]
                self.l2_grad_mean_max_min.append(mmm)

        saved_episode.append(rewards)
        my_obvs = []
        for obvs in enumerate(observations):
            my_obvs.append(obvs[1][0].numpy())

        my_log_probs = []
        for lps in enumerate(log_probs):
            my_log_probs.append(lps[1][0].data.numpy())


        saved_episode.append(my_obvs)
        saved_episode.append(self.episode_mus)
        saved_episode.append(self.episode_sigmas)
        saved_episode.append(actions)
        saved_episode.append(my_log_probs)
        saved_episode.append(loss.data.numpy())


        self.episode_sigmas = []
        self.episode_mus = []

        self.optimizer.step()

        # Now that we've updated the weights lets save them again....
        for f in self.model.linear1.parameters():
            saved_episode.append(f.data.numpy())
        for f in self.model.linear2.parameters():
            saved_episode.append(f.data.numpy())
        for f in self.model.linear2_.parameters():
            saved_episode.append(f.data.numpy())


        my_intermediates = []
        for lps in enumerate(self.intermediates):
            my_intermediates.append(lps[1][0].data.numpy())

        saved_episode.append(my_intermediates)
        saved_episode.append(returns)
        print(saved_episode[0])
        np.save("saved_episode", np.array(saved_episode))
        exit()

        return loss
