# 训练神经网络部分
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utils
import model # 导入神经网络文件
import os

BATCH_SIZE = 128
LEARNING_RATE_A = 0.001
LEARNING_RATE_C = 0.001
GAMMA = 0.99
TAU = 0.001

class Trainer:
    def __init__(self, state_dim, action_dim, a_bound, ram):
        # 初始化变量
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a_bound = a_bound
        self.ram = ram # buffer
        self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

        # Actor
        self.actor = model.Actor(self.state_dim, self.action_dim, self.a_bound) # eval net
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.a_bound)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_A) # 只更新eval net

        # Critic
        self.critic = model.Critic(self.state_dim, self.action_dim)
        self.target_critic = model.Critic(self.state_dim, self.action_dim)
        self.critc_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_C)

        # 把eval net的参数赋给target net
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        # exploitation: 动作不加噪声, 给demo用的
        # 吃state(numpy),转成tensor送入网络,再转回numpy,吐action(numpy)
        state = Variable(torch.from_numpy(state)) # numpy -> tensor
        action = self.actor.forward(state).detach() # .detach(): 取出一个分支来, 不反向更新
        return action.data.numpy() # 转回numpy

    def get_exploration_action(self, state):
        state = Variable(torch.from_numpy(state))
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() +  (self.noise.sample() * self.a_bound)
        return new_action


    def optimize(self):
        # 抽样更新网络参数
        s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)
        # 转成tensor
        s1 = Variable(torch.from_numpy(s1))
        a1 = Variable(torch.from_numpy(a1))
        r1 = Variable(torch.from_numpy(r1))
        s2 = Variable(torch.from_numpy(s2))

        # 按论文里的顺序,先更新critic,再更新actor
        ###################################### 更新critic ######################################
        # 用老网络算下一个状态s2的最优动作a2
        a2 = self.target_actor.forward(s2).detach() # target_net不更新
        q_next = torch.squeeze(self.target_critic.forward(s2, a2).detach()) # 压成一列(BATCH_SIZE, 1)
        q_target = r1 + GAMMA * q_next
        q_eval = torch.squeeze(self.critic.forward(s1, a1)) # eval_net不会fix住
        # loss_critic = F.mse_loss(q_eval, q_target)
        loss_critic = F.smooth_l1_loss(q_eval, q_target) # 两种loss,这是差的绝对值
        self.critc_optimizer.zero_grad()
        loss_critic.backward()
        self.critc_optimizer.step()

        ###################################### 更新actor #######################################
        a_pred = self.actor.forward(s1)
        loss_actor = -1*torch.sum(self.critic.forward(s1, a_pred))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # soft replacement target_net
        utils.soft_update(self.target_critic, self.critic, TAU)
        utils.soft_update(self.target_actor, self.actor, TAU)


    def save_models(self, episode, env):
        # 按环境和eps存模型,保存eval_net
        PATH = './Models_of_%s' % str(env)
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        torch.save(self.actor.state_dict(), PATH + '/' + str(episode) + '_actor.pt')
        torch.save(self.critic.state_dict(), PATH + '/' + str(episode) + '_critic.pt')
        print('Models saved successfully!!!')

    def load_models(self, episode, env):
        # 加载保存好的模型
        PATH = './Models_of_%s' % str(env)
        self.actor.load_state_dict(torch.load(PATH + '/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(PATH + '/' + str(episode) + '_critic.pt'))
        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)
        print('Models loaded successfully!!!')


