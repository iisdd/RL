import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions): # 换goal手术
        T = episode_batch['actions'].shape[1] # episode长度
        rollout_batch_size = episode_batch['actions'].shape[0]  # 多少个episode
        batch_size = batch_size_in_transitions # 抽几行,几列出来
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size) # 应该说是batch_行 吧
        # np.random.randint(low, high, size) , 包头不包尾
        t_samples = np.random.randint(T, size=batch_size) # batch_列
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()} # 把字典episode_batch中对应的行列抽出来了
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p) # future_p : 4/5 -> 正常trans和HER_trans比例为1:4
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes] # 选新goal的列
        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag  # 更新goal
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions # 只返回换了goal的trans
