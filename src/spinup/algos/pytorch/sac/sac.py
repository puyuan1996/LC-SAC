from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from .replay_buffer import ReplayBuffer
from .sampler import obtain_rollout_samples
from tensorboardX import SummaryWriter
import os
import pytorch_util as ptu


class SAC(object):
    def __init__(self, env, env_name, agent, q1_net, q2_net, seed=0,
                 steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
                 polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, seq_len=20, start_steps=10000,
                 update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
                 save_freq=10, model_path='./model/', device='cpu', train_steps=1000
                 , collect_data_samples=10000, latent_encoder_update_frequency=10, args=None):
        super().__init__()
        self.env, self.test_env = env, env
        self.env_name = env_name
        self.agent = agent
        self.q1_net = q1_net
        self.q2_net = q2_net
        self.seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr
        self.alpha = alpha
        # self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.save_freq = save_freq
        self.model_path = model_path
        self.kl_lambda = 0.1
        self.latent_batch_size = batch_size
        self.seq_len = seq_len
        self.total_train_steps = 0
        self.total_env_steps = 0
        self.duration = 0
        self.use_next_obs_in_context = False
        self.device = device
        self.train_steps = train_steps
        self.collect_data_samples = collect_data_samples
        self.update_latent_encoder = False
        self.latent_encoder_update_frequency = latent_encoder_update_frequency
        self.random_steps = args.random_steps
        self.update_begin_steps = args.update_begin_steps
        # self.sampler=sampler
        self.writer = SummaryWriter('tblogs')
        # self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]

        # Create actor-critic module and target networks

        self.q1_net_targ = deepcopy(q1_net)
        self.q2_net_targ = deepcopy(q2_net)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q1_net_targ.parameters():
            p.requires_grad = False
        for p in self.q2_net_targ.parameters():
            p.requires_grad = False
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(q1_net.parameters(), q2_net.parameters())

        # Experience buffer
        self.rl_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        # self.latent_replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [agent.latent_encoder, agent.pi, q1_net, q2_net])
        # self.logger.log('\nNumber of parameters: \t encoder:%d,\t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
        print('\nNumber of parameters: \t encoder:%d,\t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(agent.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)
        # self.latent_optimizer = Adam(agent.latent_encoder.parameters(), lr=lr)
        self.latent_optimizer = Adam(agent.latent_encoder.parameters(), lr=1e-6)

        # Set up model saving
        # logger.setup_pytorch_saver(ac)

    # def get_action(self, o, deterministic=False):
    #     with torch.no_grad():
    #         a, _ = self.agent.policy(torch.as_tensor(o, dtype=torch.float32), deterministic, False)
    #         return a.numpy()

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data, z):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o.squeeze_(0)
        o = o.to(self.device)
        a = a.to(self.device)
        o2 = o2.to(self.device)
        r = r.to(self.device)
        d = d.to(self.device)

        q1 = self.q1_net(o, a, z[0])
        q2 = self.q2_net(o, a, z[0])

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.agent.pi.get_act_logp_from_o_z(o2, z[1])

            # Target Q-values
            q1_pi_targ = self.q1_net_targ(o2, a2, z[1])
            q2_pi_targ = self.q2_net_targ(o2, a2, z[1])
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, z):
        o = data['obs']
        o.squeeze_(0)
        o = o.to(self.device)
        # pi, logp_pi = self.agent.pi(o, z)

        o_z = torch.cat([o, z], dim=1)
        pi_action, logp_pi = self.agent.pi(o_z, deterministic=False, with_logprob=True)

        q1_pi = self.q1_net(o, pi_action, z)
        q2_pi = self.q2_net(o, pi_action, z)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update_step(self, context_batch_indices, context_seq_batch, update_latent_encoder):

        data = self.rl_replay_buffer.sample_data(context_batch_indices)

        # data = data.to(self.device)
        z = self.agent(context_seq_batch.to(self.device))

        # KL constraint on z if probabilistic
        if update_latent_encoder == True:
            self.latent_optimizer.zero_grad()

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        # loss_q, q_info = self.compute_loss_q(data, z.detach())
        loss_q, q_info = self.compute_loss_q(data, z)
        loss_q.backward()
        self.q_optimizer.step()

        if update_latent_encoder == True:
            # print('before loss_q',loss_q.item())
            # print('before loss_pi',loss_pi.item())
            self.latent_optimizer.step()

        # Record things
        # self.logger.store(LossQ=loss_q.item(), **q_info)
        self.writer.add_scalar('LossQ', float(loss_q.item()), self.total_train_steps)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data, z[0].detach())
        loss_pi.backward()  # retain_graph=True
        self.pi_optimizer.step()
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        if self.total_train_steps % 2000 == 0:
            print('loss_q', loss_q.item())
            print('loss_pi', loss_pi.item())

        # Record things
        # self.logger.store(LossPi=loss_pi.item(), **pi_info)
        self.writer.add_scalar('LossPi', float(loss_pi.item()), self.total_train_steps)
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.q1_net.parameters(), self.q1_net_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.q2_net.parameters(), self.q2_net_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def sac_update_step(self):

        data = self.rl_replay_buffer.random_batch(100)

        # data = data.to(self.device)
        # z = self.agent(context_seq_batch.to(self.device))
        z = torch.zeros(2, 100, 1)
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        # loss_q, q_info = self.compute_loss_q(data, z.detach())
        loss_q, q_info = self.compute_loss_q(data, z.detach())
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        # self.logger.store(LossQ=loss_q.item(), **q_info)
        self.writer.add_scalar('LossQ', float(loss_q.item()), self.total_train_steps)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data, z[0].detach())
        loss_pi.backward()  # retain_graph=True
        self.pi_optimizer.step()
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        if self.total_train_steps % 2000 == 0:
            print('loss_q', loss_q.item())
            print('loss_pi', loss_pi.item())

        # Record things
        # self.logger.store(LossPi=loss_pi.item(), **pi_info)
        self.writer.add_scalar('LossPi', float(loss_pi.item()), self.total_train_steps)
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.q1_net.parameters(), self.q1_net_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.q2_net.parameters(), self.q2_net_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def train_step_new(self, update_latent_encoder=True):
        # self.latent_batch_size = 100
        # self.seq_len =20
        # self.agent.clear_z()
        context_seq_batch = []
        context_batch_indices = []
        # sample context batch
        for i in range(self.latent_batch_size):  # 随机采集100个长度为seq_len的sequence
            indices, data = self.rl_replay_buffer.random_sequence(seq_len=self.seq_len)  # 20
            # 采集长度为seq_len的1个sequence
            # data = data.to(self.device)
            # data = { obs，obs2, act, rew, done}
            if self.use_next_obs_in_context:
                context = torch.cat((data['obs'], data['obs2'], data['act'], data['rew'].unsqueeze(1)), dim=1)
            else:
                context = torch.cat((data['obs'], data['act'], data['rew'].unsqueeze(1)),
                                    dim=1)  # 20,15  (seq_len, feat)
            # context.unsqueeze_(0)
            # context = (obs, act, rew)

            context_seq_batch.append(context)
            context_batch_indices.append(indices[-1])  # very important 不用加1

        context_seq_batch = torch.stack(context_seq_batch)  # 100,20,15  (batch_size, seq_len, feat)
        # context_seq_batch = context_seq_batch.to(self.device)
        # (80，20，dim) dim=obs_dim + act_dim + 1      e=s,a,r
        # 比如hopper dim=11+3+1=15
        self.update_step(context_batch_indices, context_seq_batch, update_latent_encoder)  # sac rl update

        # stop backprop
        # self.agent.detach_z()

    def collect_data(self, nums_steps, add_to_latent_buffer=True, deterministic=False, random_action=False):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param nums_steps: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        # self.agent.clear_z()
        # self.rl_replay_buffer.clear()
        cur_steps = 0
        while cur_steps < nums_steps:
            paths, nums_steps = obtain_rollout_samples(self.env, self.agent, max_samples=nums_steps - cur_steps,
                                                       max_path_length=self.max_ep_len,
                                                       max_nums_paths=1000, deterministic=deterministic,
                                                       random_action=random_action)
            cur_steps += nums_steps
            self.rl_replay_buffer.add_paths(paths)
            # if add_to_latent_buffer:
            #     self.latent_replay_buffer.add_paths(paths)
        print('rew_mean:', np.array([path['rew'].sum() for path in paths]).mean())
        self.total_env_steps += cur_steps

    def train(self):
        # Prepare for interaction with environment
        total_steps = self.steps_per_epoch * self.epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        ep_ret_list = []
        hidden_in = (torch.zeros([1, 1, self.agent.hidden_dim], dtype=torch.float).to(self.agent.device),
                     torch.zeros([1, 1, self.agent.hidden_dim], dtype=torch.float).to(self.agent.device))
        while self.total_train_steps < total_steps:
            # if self.total_train_steps % 500 == 0:
            #     self.rl_replay_buffer.clear()
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if self.total_env_steps > self.random_steps:
                if ep_len == 0:
                    # a, hidden_out = agent.get_action(hidden_in, o, deterministic=deterministic)
                    hidden_out = hidden_in
                    a = self.env.action_space.sample()
                else:
                    a, hidden_out = self.agent.get_action(hidden_in, o, deterministic=False)
                hidden_in = hidden_out
                a = a.squeeze()
                # a = get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            self.total_env_steps += 1
            if self.total_env_steps > self.random_steps:
                self.agent.update_context([o, a, r, o2, d])
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            self.rl_replay_buffer.add_sample(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                ep_ret_list.append(ep_ret)
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                hidden_in = (torch.zeros([1, 1, self.agent.hidden_dim], dtype=torch.float).to(self.agent.device),
                             torch.zeros([1, 1, self.agent.hidden_dim], dtype=torch.float).to(self.agent.device))

            if self.total_env_steps % 2000 == 0:
                self.duration = time.time() - start_time
                print(f'total_env_steps:{self.total_env_steps}',
                      f'total_train_steps:{self.total_train_steps}', f'duration:{self.duration}')
                print(f'recent 5000 步 ep_ret mean max min std :', np.array(ep_ret_list).mean(),
                      np.array(ep_ret_list).max(),
                      np.array(ep_ret_list).min(),
                      np.array(ep_ret_list).std())
                ep_ret_list = []

            # Update handling
            if self.total_env_steps >= self.update_begin_steps and self.total_env_steps % 50 == 0:
                for j in range(50):
                    if self.total_train_steps % self.latent_encoder_update_frequency == 0:  # 服务器
                        self.update_latent_encoder = True
                    # self.train_step_new(self.update_latent_encoder) # TODO
                    self.sac_update_step()  #TODO
                    self.update_latent_encoder = False
                    # 从buffer中选取长度为100的transition序列，然后以20为seq_length 分成 80个sequence,作为一个batch更新
                    # (80，20，dim) dim=obs_dim + act_dim + 1      e=s,a,r
                    # 比如hopper dim=11+3+1=15
                    self.total_train_steps += 1

            if (self.total_train_steps + 1) % self.steps_per_epoch == 0:
                epoch = (self.total_train_steps + 1) // self.steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    # self.logger.save_state({'env': self.env}, None)
                    model_path = self.model_path + self.env_name + f'_s{self.seed}_{self.collect_data_samples}_{self.train_steps}/'
                    # if not os.path.exists(model_path):
                    os.makedirs(model_path, exist_ok=True)
                    torch.save(self.agent.latent_encoder.state_dict(), model_path + f'latent_encoder_{epoch}.pt')
                    torch.save(self.agent.pi.state_dict(), model_path + f'pi_{epoch}.pt')
                    torch.save(self.q1_net.state_dict(), model_path + f'q1_net_{epoch}.pt')
                    torch.save(self.q2_net.state_dict(), model_path + f'q2_net_{epoch}.pt')
