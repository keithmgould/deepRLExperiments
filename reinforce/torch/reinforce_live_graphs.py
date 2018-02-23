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
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--resume', action='store_true',
                    help='resume from saved network (default: False)')
parser.add_argument('--live_graph', action='store_true',
                    help='live graph of all states (default: False)')
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

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


policy = Policy()

if args.resume:
    policy.load_state_dict(torch.load(PATH))

optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    return probs.multinomial()

def finish_episode():
    R = 0
    rewards = []

    pdb.set_trace()

    # reverse the order, so oldest first
    for r in policy.rewards[::-1]:
        # apply the discount
        R = r + args.gamma * R
        rewards.insert(0, R)

    old_rewards = rewards
    # turn rewards into a tensor
    rewards = torch.Tensor(rewards)

    # give rewards a zero mean, and a std of 1
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # pytorch way of storing the reward on each action for later use
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)

    # zero out the gradients
    optimizer.zero_grad()

    # pdb.set_trace()

    # This function accumulates gradients in the leaves
    # http://pytorch.org/docs/master/autograd.html
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])


    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]
    torch.save(policy.state_dict(), PATH)


def print_status(i_episode):
    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, t, running_reward))

if args.live_graph:
    plt.ion()
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    xs = np.linspace(0, 100, num=100)


def graph_state(state):
    global ys
    ys = np.delete(ys,0, axis=0)
    ys = np.append(ys,[state], axis=0)
    xpos.set_ydata(ys[:,0])
    xvel.set_ydata(ys[:,1])
    theta.set_ydata(ys[:,2])
    thetavel.set_ydata(ys[:,3])
    plt.draw()
    plt.pause(0.001) # ugh

running_reward = 10
for i_episode in count(1):
    state = env.reset()
    episode_reward_sum = 0
    x_penalty_sum = 0

    if args.live_graph:
        ys = np.zeros((100,4))
        #-----------------------
        ax1.cla()
        ax1.set_title("x pos", fontsize=10)
        ax1.axis([0, 100, -2.5, 2.5])
        ax1.plot([0,100], [0,0])
        xpos = ax1.plot(xs, ys[:,0])[0]
        #-----------------------
        ax2.cla()
        ax2.set_title("x vel", fontsize=10)
        ax2.axis([0, 100, -2.5, 2.5])
        ax2.plot([0,100], [0,0])
        xvel = ax2.plot(xs, ys[:,1])[0]
        #-----------------------
        ax3.cla()
        ax3.set_title("theta pos", fontsize=10)
        ax3.axis([0, 100, -0.5, 0.5])
        ax3.plot([0,100], [0,0])
        theta = ax3.plot(xs, ys[:,2])[0]
        #-----------------------
        ax4.cla()
        ax4.set_title("theta vel", fontsize=10)
        ax4.axis([0, 100, -1, 1])
        ax4.plot([0,100], [0,0])
        thetavel = ax4.plot(xs, ys[:,3])[0]

    for t in range(1000):
        env.render() if args.render else False
        action = select_action(state)
        state, reward, done, _ = env.step(action.data[0,0])
        if args.live_graph:
            graph_state(state)
        x_penalty = 0 if state[0] == 0 else min(abs(state[0]), 1)
        reward -= x_penalty # penalize for x position
        x_penalty_sum += x_penalty
        episode_reward_sum += reward
        policy.rewards.append(reward)
        policy.saved_actions.append(action)

        if abs(state[0]) > 1:
            break

        if done:
            break
    # print("")
    print("episode reward sum: {}, t: {}, x penalty sum: {}".format(episode_reward_sum, t, x_penalty_sum))
    running_reward = running_reward * 0.99 + episode_reward_sum * 0.01
    finish_episode()
    print_status(i_episode)

    # if running_reward > env.spec.reward_threshold:
    #     print("Solved! Running reward is now {} and "
    #           "the last episode runs to {} time steps!".format(running_reward, t))
    #     break
