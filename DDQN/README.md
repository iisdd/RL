# Double DQN
也是处理离散动作的算法,与DQN不同,用eval网络挑选动作max(q_next),但是用target网络算a_next的分数,避免了overestimate
