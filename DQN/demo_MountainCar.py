# 训练&保存模型
import gym
import numpy as np
import gc # 清理内存的

import train
import buffer

ENV = 'MountainCar-v0'  # 'CartPole-v0', 'MountainCar-v0', 'BipedalWalker-v2'
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
total_steps = []
for ep in range(10):
    ep_r = 0
    ep_s = 0                                    # 每个eps花多少步能爬到山顶
    s = env.reset()

    while 1:
        env.render()
        s = np.float32(s)
        a = trainer.get_exploitation_action(s)  # 这里改成确定动作不加噪声

        s_, r, done, info = env.step(a)

        s = s_
        ep_r += r
        ep_s += 1

        if done:
            print('ep: ', ep, '  reward: %.2f' % ep_r)
            total_reward.append(ep_r)
            total_steps.append(ep_s)
            break

    gc.collect()                                # 清内存

print('demo completed')
print('平均reward: ', np.mean(total_reward))
print('平均花费%d步爬到山顶' % np.mean(total_steps))     # 500eps结果: 151步
# import matplotlib.pyplot as plt
# plt.plot(total_reward)
# plt.show()
