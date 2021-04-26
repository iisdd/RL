# 训练&保存模型
import gym
import numpy as np
import gc # 清理内存的

import train
import buffer

ENV = 'CartPole-v0'     # 'CartPole-v0', 'MountainCar-v0', 'BipedalWalker-v2'
env = gym.make(ENV)
env = env.unwrapped     # 还原env的原始设置，env外包了一层防作弊层

MAX_EPISODES = 1001
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
for ep in range(MAX_EPISODES):
    ep_r = 0
    s = env.reset()
    # if ep > MAX_EPISODES - 10: RENDER = True

    while 1:
        if RENDER: env.render()
        s = np.float32(s)
        a = trainer.get_exploration_action(s)

        s_, r, done, info = env.step(a)

        # 修改一下reward,carpole专用
        x, x_dot, theta, theta_dot = s_  # 这里的 N_STATES包括4个特征
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        if done:
            next_state = None
        else:
            next_state = np.float32(s_)
            ram.add(s, a, r, next_state)

        s = s_
        ep_r += r
        trainer.optimize()
        if done or ep_r > 1000:     # reward能有1000就可以杀了
            print('ep: ', ep, '  reward: %.2f' % ep_r)
            total_reward.append(ep_r)
            break

    gc.collect() # 清内存

    if ep % 100 == 0:
        trainer.save_models(episode=ep, env=ENV)

print('training completed')

import matplotlib.pyplot as plt
plt.plot(total_reward)
plt.show()
print('后100eps平均rewards: ', np.mean(total_reward[-100: ]))
