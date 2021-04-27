# 训练&保存模型,环境: 月球停车
# reward = 200 算解决
# 降落得分100~140(根据完成时间和落地时的速度决定), 成功降落+100,坠毁-100, 每条腿着陆+10, 每帧喷气-0.3
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import gc # 清理内存的

import train
import buffer

ENV = 'LunarLanderContinuous-v2' # 'Pendulum-v0' , 'BipedalWalker-v3', 'LunarLanderContinuous-v2'
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
trainer.load_models(episode=1079, env=ENV)

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