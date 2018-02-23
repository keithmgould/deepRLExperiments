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

        # for f in self.model.linear1.parameters():
        #     pdb.set_trace()

        # self.model = self.model
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

    # probability density of x given a normal distribution
    # defined by mu and sigma
    def normal(self, x, mu, sigma_sq):
        a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
        b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
        return a*b

    def select_action(self, state):
        state = Variable(state)
        mu, sigma_sq = self.model(state)
        # foo = make_dot(mu, params=dict(self.model.named_parameters()))
        # pdb.set_trace()

        # random scalar from normal distribution
        # with mean 0 and std 1
        random_from_normal = torch.randn(1)

        # modulate our normal (mu,sigma) with random_from_normal to pick an action.
        # Note that if x = random_from_normal, then our action is just:
        # mu + sigma * x
        sigma = sigma_sq.sqrt()
        self.episode_sigmas.append(sigma.data[0][0])
        self.episode_mus.append(mu.data[0][0])
        action = (mu + sigma*Variable(random_from_normal)).data

        # calculate the probability density
        prob = self.normal(action, mu, sigma_sq)

        log_prob = prob.log()

        return action, log_prob

    def discount_rewards(self, rewards, gamma):
        stepReturn = 0
        stepReturns = []
        for i in range(len(rewards)):
            stepReturn = gamma * stepReturn + rewards[i]
            stepReturns.append(stepReturn)
        return list(reversed(stepReturns))

    def update_parameters(self, rewards, log_probs, gamma):
        discounted_rewards = self.discount_rewards(rewards, gamma)
        loss = 0

        for i in range(len(rewards)):
            foo = log_probs[i]*discounted_rewards[i]
            loss = loss + foo[0]
        loss = loss / len(rewards)
        loss = -loss
        self.losses.append(loss.data[0])
        self.ep_len.append(len(rewards))
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
                # print("L1 grad nonzero: {}".format(bar))
        for f in self.model.linear2_.parameters():
            if(np.size(f.data.numpy()) == 128):
                foo = f.grad.data.numpy()
                bar = np.count_nonzero(foo)
                # print("L2 grad nonzero: {}".format(bar))
                self.l2_nz.append(bar)
                mmm = [np.mean(foo), np.max(foo), np.min(foo)]
                self.l2_grad_mean_max_min.append(mmm)

        self.optimizer.step()
        # l1 = self.model.linear1.weight.data.numpy()
        # l2 = self.model.linear2.weight.data.numpy()
        # l1_nz = np.array(self.l1_nz)
        # l2_nz = np.array(self.l2_nz)
        # print("L1NZ: mean: {}, std: {}".format(np.mean(l1_nz), np.std(l1_nz)))
        # print("L2NZ: mean: {}, std: {}".format(np.mean(l2_nz), np.std(l2_nz)))
        # print("L1 W: {}, {}, {}".format(np.min(l1),np.max(l1),np.mean(l1)))
        # print("L2 W: {}, {}, {}".format(np.min(l2),np.max(l2),np.mean(l2)))
        return loss
