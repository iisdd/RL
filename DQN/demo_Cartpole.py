# 训练&保存模型
import gym
import numpy as np
import gc # 清理内存的

import train
import buffer

ENV = 'CartPole-v0'     # 'CartPole-v0', 'MountainCar-v0', 'BipedalWalker-v2'
env = gym.make(ENV)
env = env.unwrapped     # 还原env的原始设置，env外包了一层防作弊层

MAX_BUFFER = 10000

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.n


print(' Env: ', ENV)
print(' State Dimension: ', S_DIM)
print(' Number of Action(discrete) : ', A_DIM)


ram = buffer.ReplayBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, ram)
trainer.load_models(episode=500, env=ENV)


total_reward = []
for ep in range(10):
    ep_r = 0
    s = env.reset()

    while 1:
        # env.render()                          # 太费时间了...
        s = np.float32(s)
        a = trainer.get_exploitation_action(s)  # 这里改成确定动作不加噪声

        s_, r, done, info = env.step(a)

        # 修改一下reward
        x, x_dot, theta, theta_dot = s_         # 这里的 N_STATES包括4个特征
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        s = s_
        ep_r += r

        if done or ep_r > 10000:                # 我们认为 > 10000就是死不掉吧
            print('ep: ', ep, '  reward: %.2f' % ep_r)
            total_reward.append(ep_r)
            break

    gc.collect()                                # 清内存

print('demo completed')
print('平均reward: ', np.mean(total_reward))
import matplotlib.pyplot as plt
plt.plot(total_reward)
plt.show()
