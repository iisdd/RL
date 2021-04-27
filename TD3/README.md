# TD3
训练的环境都要在linux环境下加载(要用到mujoco)

用于连续动作的算法,包括六张网,两个critic一个actor以及他们的target net

选a'时用a_targ,并且加噪声,每次算q_target,选两个c_targ里面比较小q的作q_next,参数滑动更新,C更新d(Delayed)次,A更新一次
