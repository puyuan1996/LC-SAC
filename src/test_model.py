import os
import torch
import torch.nn as nn

from networks import MlpEncoder, RecurrentEncoder
from agent import Agent

import gym
from spinup.algos.pytorch.sac.core import SquashedGaussianMLPActor, MLPQFunction
from spinup.algos.pytorch.sac.sac import SAC
import pytorch_util as ptu
import time
import numpy as np


def run(args):
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    reward_dim = 1

    # instantiate networks
    latent_dim = args.latent_dim
    latent_encoder_input_dim = 2 * obs_dim + act_dim + reward_dim if args.use_next_obs_in_context else obs_dim + act_dim + reward_dim
    latent_encoder_output_dim = latent_dim * 2
    recurrent = True
    latent_encoder_hidden_dim = 200
    if recurrent:
        latent_encoder = RecurrentEncoder(input_dim=latent_encoder_input_dim, latent_dim=5,
                                          hidden_dim=latent_encoder_hidden_dim)
    else:
        latent_encoder = MlpEncoder(
            hidden_sizes=(200, 200, 200),
            input_size=latent_encoder_input_dim,
            output_size=latent_encoder_output_dim,
        )

    # hidden_sizes = (net_size, net_size, net_size)
    hidden_sizes = (300, 300, 300)
    activation = nn.ReLU
    # build policy and value functions
    policy = SquashedGaussianMLPActor(obs_dim + latent_dim, act_dim, hidden_sizes, activation, act_limit)
    q1_net = MLPQFunction(obs_dim + latent_dim, act_dim, hidden_sizes, activation)
    q2_net = MLPQFunction(obs_dim + latent_dim, act_dim, hidden_sizes, activation)
    agent = Agent(latent_dim, latent_encoder_hidden_dim, latent_encoder, policy, args)

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    algorithm = SAC(env, agent, q1_net, q2_net,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs, max_ep_len=args.max_ep_len,
                    save_freq=args.save_freq, model_path=args.model_path,
                    logger_kwargs=logger_kwargs)

    # optionally load pre-trained weights
    if args.load_model:
        model_path = args.model_path
        latent_encoder.load_state_dict(torch.load(os.path.join(model_path, 'latent_encoder.pth')))
        q1_net.load_state_dict(torch.load(os.path.join(model_path, 'q1_net.pth')))
        q2_net.load_state_dict(torch.load(os.path.join(model_path, 'q2_net.pth')))
        policy.load_state_dict(torch.load(os.path.join(model_path, 'pi.pth')))
        print('load model successfully')

    print(obs_dim, act_dim)
    test_eps = 10
    accum_context = True
    rewards = 0
    path_length = 0
    num_eps = 0
    total_rewards = []
    o = env.reset()
    agent.clear_z()
    hidden_in = (torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(agent.device),
                 torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(agent.device))
    while num_eps < test_eps:
        if path_length == 0:
            # a, hidden_out = agent.get_action(hidden_in, o, deterministic=deterministic)
            hidden_out = hidden_in
            a = env.action_space.sample()
        else:
            a, hidden_out = agent.get_action(hidden_in, o, deterministic=True)
        hidden_in = hidden_out
        a = a.squeeze()
        next_o, r, d, env_info = env.step(a)
        agent.update_context([o, a, r, next_o, d])
        env.render()
        # time.sleep(0.02)
        rewards += r
        path_length += 1
        o = next_o

        if d:
            num_eps += 1
            print(f'rewards:{rewards},ave rewards:{rewards / path_length},path_length:{path_length},')
            time.sleep(0.2)
            total_rewards.append(rewards)
            rewards = 0
            path_length = 0
            env.seed(num_eps)
            o = env.reset()
            agent.clear_z()

    total_rewards = np.array(total_rewards)
    print( f'total_rewards\n mean: {total_rewards.mean()},std: {total_rewards.std()},max: {total_rewards.max()},min: {total_rewards.min()}')


env_id_dict = {0: 'Ant-v2', 1: 'HalfCheetah-v2', 2: 'Hopper-v2', 3: 'Humanoid-v2',
               4: 'Pusher-v2', 5: 'Reacher-v2', 6: 'Striker-v2', 7: 'Swimmer-v2', 8: 'Thrower-v2',
               9: 'LunarLanderContinuous-v2', 10: 'BipedalWalkerHardcore-v3'}
learner_path_dict = {0: 'ant', 1: 'halfcheetah', 2: 'hopper', 3: 'humanoid',
                     4: 'pusher', 5: 'reacher', 6: 'striker', 7: 'swimmer', 8: 'thrower'}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        default='LunarLanderContinuous-v2')  # BipedalWalker-v3' MountainCarContinuous-v0' HalfCheetah-v2'
    parser.add_argument('--max_ep_len', type=int, default=10000)  # 默认是1000
    parser.add_argument('--save_freq', type=int, default=50)  # 默认是1
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.9995)  # 0.995
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)  # 50*4000=2e5
    parser.add_argument('--exp_name', type=str, default='lc-sac')
    parser.add_argument('--model_path', type=str, default='./model/')
    parser.add_argument('--load_model', type=bool, default=True)

    parser.add_argument('--latent_dim', type=int, default=5)
    # parser.add_argument('--base_log_dir', type=str, default='./log')
    parser.add_argument('--use_next_obs_in_context', type=bool, default=False)
    parser.add_argument('--recurrent', type=bool, default=True)
    args = parser.parse_args()

    for j in range(5):
        for i in [2]:
            torch.manual_seed(j)
            args.env = env_id_dict[i]
            # args.base_log_dir = './log'
            args.use_next_obs_in_context = False

            if i == 2:  # Hopper
                args.epochs = 500  # =1e6
            elif i in [0, 1]:  # Ant,HalfCheetah
                args.epochs = 500  # 3e6
            elif i == 3:  # Humanoid
                args.epochs = 1000  # 10e6
            else:  # Striker,Pusher,Thrower
                args.epochs = 1000  # 4e6
            run(args)
