# 训练&保存模型
import gym
import numpy as np
import gc # 清理内存的

import train
import buffer

ENV = 'MountainCar-v0'  # 'CartPole-v0', 'MountainCar-v0', 'BipedalWalker-v2'
env = gym.make(ENV)
env = env.unwrapped     # 还原env的原始设置，env外包了一层防作弊层

MAX_EPISODES = 501
MAX_BUFFER = 10000

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.n


print(' Env: ', ENV)
print(' State Dimension: ', S_DIM)
print(' Number of Action(discrete) : ', A_DIM)


ram = buffer.ReplayBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, ram)

RENDER = False

total_reward = []
total_step = []     # 记录每个eps多少steps能搞定
for ep in range(MAX_EPISODES):
    ep_r = 0
    ep_steps = 0
    s = env.reset()
    if ep > MAX_EPISODES - 10: RENDER = True

    while 1:
        if RENDER: env.render()
        s = np.float32(s)
        a = trainer.get_exploration_action(s)

        s_, r, done, info = env.step(a)
        # MountainCar专用reward
        position , velocity = s_
        r = abs(position - (-0.5)) - 0.5 # 奖励的距离让它左右摆起来, -0.5让它快点搞定, 同时为了reward收敛到0

        if done:
            r = 10 # mountaincar专用
            next_state = None
        else:
            next_state = np.float32(s_)
            ram.add(s, a, r, next_state)

        s = s_
        ep_r += r
        ep_steps += 1
        trainer.optimize()
        if done:
            print('ep: ', ep, '  reward: %.2f' % ep_r)
            total_reward.append(ep_r)
            total_step.append(ep_steps)
            break

    gc.collect() # 清内存

    if ep % 100 == 0:
        trainer.save_models(episode=ep, env=ENV)

print('training completed')

import matplotlib.pyplot as plt
plt.figure()
plt.plot(total_reward)
plt.title('reward_curve')
plt.show()
plt.figure()
plt.plot(total_step)
plt.title('steps_curve')
plt.show()
