# 一些接口工具,方法
import numpy as np
import torch
import shutil # 文件处理包
import torch.autograd as Variable

def soft_update(target, source, tau):
	# soft replacement : 用source(新)更新target(旧),注意要用.data来进行值更新
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)


def hard_update(target, source):
	# 相当于赋值,assign
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)



# 加噪声
class OrnsteinUhlenbeckActionNoise:
	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu
		# print('初始化X') # 这只初始化一次

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		# print(self.X, self.mu) # 这边X每次都在变
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X

# 画下图看下噪声分布
if __name__ == '__main__':
	ou = OrnsteinUhlenbeckActionNoise(1)
	states = []
	for i in range(1000):
		states.append(ou.sample())
	import matplotlib.pyplot as plt
	plt.plot(states)
	plt.show()


