# 建立神经网络,改网络参数就在这改
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

class NET(torch.nn.Module):                     # 定义神经网络
    def __init__(self, state_dim, action_dim):
        # 输出每个动作的q(s, a)
        super(NET, self).__init__()
        self.fc1 = nn.Linear(state_dim, 50)     # 在这改网络结构
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2 = nn.Linear(50, action_dim)
        self.fc2.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):                   # [m, state_dim] -> [m, action_dim](q值表)
        x = F.relu(self.fc1(state))
        action_values = self.fc2(x)             # 输出层不激活
        return action_values                    # 返回所有a的q