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
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层

MAX_EPISODES = 10001
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

total_reward = []
best_r = -100000
for ep in range(MAX_EPISODES):
    ep_r = 0
    s = env.reset()
    for _ in range(MAX_STEPS):
        s = np.float32(s)
        a = trainer.get_exploration_action(s)

        s_, r, done, info = env.step(a)

        if done:
            next_state = None
        else:
            next_state = np.float32(s_)
            ram.add(s, a, r, next_state)

        s = s_
        ep_r += r
        trainer.optimize()

        if done:
            break
    total_reward.append(ep_r)
    print('ep: ', ep, '  reward: %.2f' % ep_r)
    gc.collect() # 清内存

    if ep_r > best_r:
        best_r = ep_r
        print('ep: ', ep, '新最佳: ', best_r)
        trainer.save_models(episode=ep, env=ENV)

print('training completed')
import matplotlib.pyplot as plt
plt.figure()
plt.plot(total_reward)
plt.title('reward_curve')
plt.show()


