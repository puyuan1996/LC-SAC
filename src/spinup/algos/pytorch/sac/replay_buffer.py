from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.clear()

    def terminate_episode(self):
        self.episode_starts.append(self.cur_episode_starts)
        self.cur_episode_starts = self.ptr

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.episode_starts = []
        self.cur_episode_starts = 0

    def add_sample(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if done==True:  # TODO
            self.terminate_episode()

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (obs, act, rew, obs2, done) in enumerate(zip(
                path["obs"],
                path["act"],
                path["rew"],
                path["obs2"],
                path["done"],
        )):
            self.add_sample(obs, act, rew, obs2, done)
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def sample_data(self, idxs):
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        # indices = np.random.randint(0, self.size, batch_size)
        indices = np.random.randint(1, self.size-1, batch_size) # TODO
        return self.sample_data(indices)

    def random_sequence(self, seq_len=20):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < seq_len:
            # 采集连续的长度为 seq_len 的sequence，如果采集的一局长度小于seq_len则重新采样
            if i < 3:  # 如果采集2次都没有找到一局长度大于seq_len,则可以从几局拼接成seq_len=20
                indices = []
            # else:
            #     print(f"can not sample one episode whose seq_len>{seq_len} in {i} tries")
            #     # break
            start = np.random.choice(self.episode_starts[:-1])  # 随机采样episode开始位置
            pos_idx = self.episode_starts.index(start)
            indices += list(range(start, self.episode_starts[pos_idx + 1]))
            i += 1

        # cut off the last traj if needed to respect batch size
        indices = indices[:seq_len]

        return indices, self.sample_data(indices)
