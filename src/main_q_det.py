# -*- coding: utf-8 -*
import os
import torch
import torch.nn as nn

from latent_encoder import RecurrentLatentEncoder, RecurrentLatentEncoderSmall,  RecurrentLatentEncoderDet
from agent_det import Agent

import gym
from spinup.algos.pytorch.sac.core import SquashedGaussianMLPActor, MLPQFunction
from spinup.algos.pytorch.sac.sac_q_det_v2 import SAC
import pytorch_util as ptu


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
    latent_encoder_hidden_dim = args.latent_encoder_hidden_dim  # 128  # 128
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = 'cpu'  # device

    if recurrent:
        latent_encoder = RecurrentLatentEncoderDet(input_dim=latent_encoder_input_dim, latent_dim=latent_dim,
                                                     hidden_dim=latent_encoder_hidden_dim, device=args.device)
    else:
        latent_encoder = MlpEncoder(
            hidden_sizes=(200, 200, 200),
            input_size=latent_encoder_input_dim,
            output_size=latent_encoder_output_dim,
        )

    # hidden_sizes = (net_size, net_size, net_size)
    hidden_sizes = (256, 256)
    activation = nn.ReLU

    policy = SquashedGaussianMLPActor(obs_dim + latent_dim, act_dim, hidden_sizes, activation, act_limit).to(
        args.device)
    q1_net = MLPQFunction(obs_dim + latent_dim, act_dim, hidden_sizes, activation).to(args.device)
    q2_net = MLPQFunction(obs_dim + latent_dim, act_dim, hidden_sizes, activation).to(args.device)
    agent = Agent(latent_dim, latent_encoder_hidden_dim, latent_encoder, policy, args).to(args.device)

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    algorithm = SAC(env, args.env, agent, q1_net, q2_net,
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs, max_ep_len=args.max_ep_len,
                    seq_len=args.seq_len, lr=args.sac_lr, batch_size=args.latent_batch_size,
                    save_freq=args.save_freq, model_path=args.model_path, device=args.device,
                    train_steps=args.train_steps, collect_data_samples=args.collect_data_samples,
                    latent_encoder_update_frequency=args.latent_encoder_update_frequency,
                    args=args)

    # optionally load pre-trained weights
    if args.load_model:
        model_path = args.model_path
        latent_encoder.load_state_dict(torch.load(os.path.join(model_path, 'latent_encoder.pth')))
        q1_net.load_state_dict(torch.load(os.path.join(model_path, 'q1_net.pth')))
        q2_net.load_state_dict(torch.load(os.path.join(model_path, 'q2_net.pth')))
        policy.load_state_dict(torch.load(os.path.join(model_path, 'pi.pth')))
        print('load model successfully')

    # run the algorithm
    algorithm.train()


env_id_dict = {0: 'Ant-v2', 1: 'HalfCheetah-v2', 2: 'Hopper-v2', 3: 'Humanoid-v2',
               4: 'Pusher-v2', 5: 'Reacher-v2', 6: 'Striker-v2', 7: 'Swimmer-v2', 8: 'Thrower-v2',
               9: 'LunarLanderContinuous-v2', 10: 'BipedalWalker-v3', 11: 'BipedalWalkerHardcore-v3'}
learner_path_dict = {0: 'ant', 1: 'halfcheetah', 2: 'hopper', 3: 'humanoid',
                     4: 'pusher', 5: 'reacher', 6: 'striker', 7: 'swimmer', 8: 'thrower'}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        default='Hopper-v2')#Striker-v2')  # BipedalWalker-v3' MountainCarContinuous-v0' HalfCheetah-v2'
    parser.add_argument('--max_ep_len', type=int, default=10000)  # 默认是1000
    parser.add_argument('--save_freq', type=int, default=50)  # 默认是1
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)  # 0.995
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)  # 500*4000=2e6
    parser.add_argument('--exp_name', type=str, default='lc-sac')
    parser.add_argument('--model_path', type=str, default='./model_q_det/')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--latent_dim', type=int, default=5)
    parser.add_argument('-latent_hd', '--latent_encoder_hidden_dim', type=int, default=128)#64)
    # parser.add_argument('--base_log_dir', type=str, default='./log')
    parser.add_argument('--use_next_obs_in_context', type=bool, default=False)
    parser.add_argument('--recurrent', type=bool, default=True)
    parser.add_argument('--train_steps', type=int, default=50)
    parser.add_argument('--collect_data_samples', type=int, default=50)
    parser.add_argument('--random_steps', type=int, default=10000)  # 10000
    parser.add_argument('--update_begin_steps', type=int, default=2000)  # 1000
    parser.add_argument('-latent_uf', '--latent_encoder_update_frequency', type=int, default=5000)  # 5000 TODO
    parser.add_argument('-rl_uf', '--rl_update_frequency', type=int, default=1000)  # 1e4
    parser.add_argument('--kl_lambda', type=float, default=0.1)

    parser.add_argument('--latent_lr', type=float, default=1e-4)#1e-6
    parser.add_argument('--sac_lr', type=float, default=1e-3)  # 1e-3)
    parser.add_argument('--latent_buffer_size', type=int, default=1000000) # 50000
    parser.add_argument('--rl_buffer_size', type=int, default=5000) # 5000  TODO
    # 100 1e-3
    # 1 1e-6
    parser.add_argument('--seq_len', type=int, default=8)#20)
    parser.add_argument('--latent_batch_size', type=int, default=100)
    parser.add_argument('--env_id_str', type=list, default=[2, 1, 0] + list(range(3, 12)))
    parser.add_argument('--deterministic', type=bool, default=False)
    args = parser.parse_args()


    def get_key(dict, value):
        return [k for k, v in dict.items() if v == value]


    for j in range(5):
        torch.manual_seed(j)
        args.use_next_obs_in_context = False
        i = get_key(env_id_dict, args.env)[0]
        if i == 2:  # Hopper
            args.epochs = 1000  # 4e6
        elif i in [0, 1, 7, 9]:  # Ant,HalfCheetah, Swimmer,LunarLanderContinuous
            args.epochs = 1000  # 4e6
        elif i in [3, 11]:  # Humanoid,BipedalWalkerHardCore
            args.epochs = 2500  # 10e6
        else:  # Striker,Pusher,Reacher,Thrower, BipedalWalker
            args.epochs = 1000  # 4e6
        print(f'experiment {args.env} seed {j} begin!')
        run(args)
        print(f'experiment {args.env} seed {j} done!')
    print(f'experiment {args.env} all seed done!')
