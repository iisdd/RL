# 训练&保存模型,环境: 紫色机器人走路
# reward = 300 算解决
# 从头走到尾一共300分距离分,摔倒-100,使用电机也要消耗一些分数
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc # 清理内存的

import train
import buffer

ENV = 'BipedalWalker-v3' # 'Pendulum-v0', 'BipedalWalker-v3', 'LunarLanderContinuous-v2'
env = gym.make(ENV)

MAX_STEPS = 1000
MAX_BUFFER = 1000000

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_BOUND = env.action_space.high[0]

print(' Env: ', ENV)
print(' State Dimension: ', S_DIM)
print(' Action Dimension: ', A_DIM)
print(' Action Bound: ', A_BOUND)

ram = buffer.ReplayBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_BOUND, ram)
trainer.load_models(episode=909, env=ENV)

total_reward = []
for ep in range(10): # 展示10个回合
    ep_r = 0
    s = env.reset()
    for _ in range(MAX_STEPS):
        env.render()
        s = np.float32(s)
        a = trainer.get_exploitation_action(s)  # 这里改成确定动作不加噪声

        s_, r, done, info = env.step(a)
        # r /= 10                                 # reward除以了10
        s = s_
        ep_r += r
        if done:
            # print('ep: ', ep, '  reward: %.2f' % ep_r)
            total_reward.append(ep_r)
            break
    print('ep: ', ep, '  reward: %.2f' % ep_r)
    gc.collect() # 清内存


print('demo completed')
import matplotlib.pyplot as plt
plt.plot(total_reward)