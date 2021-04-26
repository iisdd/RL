# 一些接口工具,方法,原始的DQN用的硬替换,直接隔多少步把eval_net赋给targ_net
import numpy as np
import torch

def hard_update(target, source):
	# 更新target net
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)