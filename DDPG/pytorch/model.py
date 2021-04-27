# 建立神经网络,改网络参数就在这改
# .weight.data.size(), 少打了一个.data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003 # 输出层初始化的值

def fanin_init(size, fanin=None):
    # 高级初始化,size指上一层神经元的数量
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        # s先过两层,a过一层,两个汇合再过两层FC
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # s层1
        self.fcs1 = nn.Linear(state_dim, 256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        # s层2
        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        # a层1
        self.fca1 =nn.Linear(action_dim, 128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        # 汇合FC1
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        # 汇合FC2
        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        # 输入state,action,输出Q(s, a)
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层不激活

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, a_bound):
        # 输入s,输出a:(-1,1) * action_lim
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a_bound = a_bound
        # FC1
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        # FC2
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        # FC3
        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        # FC4
        self.fc4 = nn.Linear(64, action_dim)
        self.fc4.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):
        # 输入s,输出a
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.tanh(self.fc4(x)) # 用tanh框在(-1, 1)

        action = x * self.a_bound
        return action
#
#
#
# # Pendulum-v0专用网络!!!
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         # s先过两层,a过一层,两个汇合再过两层FC
#         super(Critic, self).__init__()
#
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#
#         # s层1
#         self.fcs1 = nn.Linear(state_dim, 30)
#         self.fcs1.weight.data = fanin_init(self.fcs1.weight.size())
#
#         # a层1
#         self.fca1 =nn.Linear(action_dim, 30)
#         self.fca1.weight.data = fanin_init(self.fca1.weight.size())
#
#         # 汇合FC1
#         self.fc2 = nn.Linear(30, 1)
#         self.fc2.weight.data.uniform_(-EPS, EPS)
#
#
#
#     def forward(self, state, action):
#         # 输入state,action,输出Q(s, a)
#         s1 = F.relu(self.fcs1(state))
#         a1 = F.relu(self.fca1(action))
#         x = s1 + a1
#
#         x = self.fc2(x)
#
#         return x
#
#
# class Actor(nn.Module): # 单层[s_dim, 30, a_dim]
#     def __init__(self, state_dim, action_dim, a_bound):
#         # 输入s,输出a:(-1,1) * action_lim
#         super(Actor, self).__init__()
#         self.a_bound = a_bound
#         # FC1
#         self.fc1 = nn.Linear(state_dim, 30)
#         self.fc1.weight.data = fanin_init(self.fc1.weight.size())
#         # FC2
#         self.fc2 = nn.Linear(30, action_dim)
#         self.fc2.weight.data.uniform_(-EPS, EPS)
#
#
#
#     def forward(self, state):
#         # 输入s,输出a
#         x = F.relu(self.fc1(state))
#         x = F.tanh(self.fc2(x)) # 用tanh框在(-1, 1)
#
#         action = x * self.a_bound
#         return action


