import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class Agent(nn.Module):

    def __init__(self,
                 latent_dim,
                 hidden_dim,
                 latent_encoder,
                 policy,
                 args
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.latent_encoder = latent_encoder
        self.pi = policy
        self.device = args.device
        self.seq_len = args.seq_len
        self.latent_batch_size = args.latent_batch_size

        self.recurrent = args.recurrent
        self.use_ib = True
        self.use_next_obs_in_context = args.use_next_obs_in_context

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(1, self.latent_dim)
        if self.use_ib:
            log_std = ptu.ones(1, self.latent_dim)
        else:
            log_std = ptu.zeros(1, self.latent_dim)
        self.z_mu = mu
        # self.z_log_std = log_std
        # sample a new z from the prior
        # self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        # self.latent_encoder.reset(1)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        # if self.recurrent:
        #     self.latent_encoder.hidden = self.latent_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''

        o, a, r, no, d = [torch.as_tensor(v, dtype=torch.float32) for v in inputs]

        r.unsqueeze_(0)

        # o = ptu.from_numpy(o[None, None, ...])
        # a = ptu.from_numpy(a[None, None, ...])
        # r = ptu.from_numpy(np.array([r])[None, None, ...])
        # no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=-1)
            data.unsqueeze_(0)

        else:
            # data = torch.cat([o, a, r], dim=2)
            data = torch.cat([o, a, r], dim=-1)
            data.unsqueeze_(0)

        self.context = data

        # if self.context is None:
        #     self.context = data
        # else:
        #     self.context = torch.cat([self.context, data], dim=0)

    def reparameterize(self, mu, log_std):
        # std = torch.exp(0.5 * log_var)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std
        # return torch.tanh(mu + eps * std)  # TODO

    def sample_z(self):
        self.z = self.reparameterize(self.z_mu, self.z_log_std)

    def get_action(self, hidden_in, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        # hidden_in = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).to(self.device),
        #              torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).to(self.device))
        params, hidden_out = self.latent_encoder(self.context.unsqueeze_(0), hidden_in)
        # params = params[:, -1, :]
        """recurrent latent encoder"""
        self.z_mu = params.view(-1, self.latent_dim)

        self.z = self.z_mu  # 均值 TODO
        # self.z = torch.zeros(1, self.latent_dim)  # TODO

        obs = ptu.from_numpy(obs[None])
        o_z = torch.cat([obs, self.z], dim=1)
        return self.pi.get_action_from_cat_o_z(o_z, deterministic=deterministic), hidden_out

    def infer_latent(self, context_seq_batch):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        hidden_in = (torch.zeros([1, context_seq_batch.shape[0], self.hidden_dim], dtype=torch.float).to(self.device),
                     torch.zeros([1, context_seq_batch.shape[0], self.hidden_dim], dtype=torch.float).to(self.device))
        params, _ = self.latent_encoder(context_seq_batch, hidden_in)
        # params 包含z_t 和z_t+1的参数

        self.z_mu = params.view(-1, self.latent_dim)  # 100,2,5 -> 200,5

    def forward(self, context_seq_batch):
        self.infer_latent(context_seq_batch)
        self.z = torch.stack((torch.stack([self.z_mu[2 * i] for i in range(self.latent_batch_size)]),
                              torch.stack([self.z_mu[2 * i + 1] for i in range(self.latent_batch_size)])))
        # self.z_mu # TODO
        # self.z = torch.zeros(2, 100, self.latent_dim)  # TODO

        return self.z  # 均值

    @property
    def networks(self):
        return [self.latent_encoder, self.pi]
