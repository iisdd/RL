# DDPG
处理连续动作的算法,actor吃s吐a(-1,1),在乘一个action_bound,critic吃a,s吐Q(a,s)

actor朝着让Q(s,a)更大的方向更新,critic朝着估计Q(s,a)更准的方向更新

一共四张网,两张actor两张critic, a_targ用来选s'的a', c_targ用来算q_next:Q(s',a') -> q_target,网络的更新用的软替换(指数平滑)
