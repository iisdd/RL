# replaybuffer模块,用的双向队列,装满后append自动pop掉第一个,每一个元素就是一个transition
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxsize = size
        self.len = 0 # 记录实时长度

    def sample(self, batch_size):
        # 抽batch训练,sample出来的是numpy.arr
        batch_size = min(batch_size, self.len)
        batch = random.sample(self.buffer, batch_size)

        s_arr = np.float32([arr[0] for arr in batch])
        a_arr = np.float32([arr[1] for arr in batch])
        r_arr = np.float32([arr[2] for arr in batch])
        s1_arr = np.float32([arr[3] for arr in batch])

        return s_arr, a_arr, r_arr, s1_arr

    def len(self):
        return self.len

    def add(self, s, a, r, s1):
        # 添加新transition
        transition = (s, a, r, s1)
        self.len += 1
        if self.len > self.maxsize:
            self.len = self.maxsize
        self.buffer.append(transition)