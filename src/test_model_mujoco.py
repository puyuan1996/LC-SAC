import os
import torch
import torch.nn as nn

from latent_encoder import RecurrentLatentEncoder, RecurrentLatentEncoder2head, RecurrentLatentEncoderDet
from agent_v3 import Agent
import gym
from spinup.algos.pytorch.sac.core import SquashedGaussianMLPActor, MLPQFunction
from spinup.algos.pytorch.sac.sac_cpl_sar_mujoco import SAC
import pytorch_util as ptu
import time
import numpy as np


def run(args):
    env = gym.make(args.env)
    import metaworld
    import random
    # ml1 = metaworld.ML1(f'{args.env}')  # Construct the benchmark, sampling tasks
    # env = ml1.train_classes[f'{args.env}']()  # Create an environment with task `pick_place`
    # task = random.choice(ml1.train_tasks)  # ml1.train_tasks 长度为50的list
    # env.set_task(task)  # Set task 这里的env只是用来求观察空间和动作空间

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
    torch.cuda.set_device(args.cuda_id)  # id=0, 1, 2 ,4等 TODO
    print(f'cuda_id:{args.cuda_id}')

    args.device = device  # 'cpu'
    print('-' * 10)
    print(f'device: {args.device}')
    print('-' * 10)

    if recurrent:
        latent_encoder = RecurrentLatentEncoder2head(input_dim=latent_encoder_input_dim, latent_dim=latent_dim,
                                                     hidden_dim=latent_encoder_hidden_dim, device=args.device)

    # hidden_sizes = (net_size, net_size, net_size)
    hidden_sizes = (256, 256)
    activation = nn.ReLU

    policy = SquashedGaussianMLPActor(obs_dim + latent_dim, act_dim, hidden_sizes, activation, act_limit).to(
        args.device)
    q1_net = MLPQFunction(obs_dim + latent_dim, act_dim, hidden_sizes, activation).to(args.device)  # MLP
    q2_net = MLPQFunction(obs_dim + latent_dim, act_dim, hidden_sizes, activation).to(args.device)
    agent = Agent(latent_dim, latent_encoder_hidden_dim, latent_encoder, policy, args).to(args.device)

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    algorithm = SAC(env, args.env, agent, q1_net, q2_net,
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs, max_ep_len=args.max_ep_len,
                    seq_len=args.seq_len, lr=args.sac_lr,
                    save_freq=args.save_freq, model_path=args.model_path, device=args.device,
                    train_steps=args.train_steps, collect_data_samples=args.collect_data_samples,
                    # latent_encoder_update_every=args.latent_encoder_update_every,
                    args=args)

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
                        default='Humanoid-v2') #LunarLanderContinuous-v2')  # BipedalWalker-v3' MountainCarContinuous-v0' HalfCheetah-v2'
    parser.add_argument('--max_ep_len', type=int, default=1000)  # TODO默认是1000
    parser.add_argument('--save_freq', type=int, default=50)  # 单位为epoch, 默认是1
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--polyak', type=float, default=0.995)  # 0.995
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)  # 500*4000=2e6
    parser.add_argument('--steps_per_epoch', type=int, default=10000)  # 500*1e4=5e6
    parser.add_argument('--exp_name', type=str, default='lc-sac')
    parser.add_argument('--model_path', type=str, default='./model_q/')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--no_cuda', type=bool, default=False)

    parser.add_argument('-latent_hd', '--latent_encoder_hidden_dim', type=int, default=128)  # 64)
    # parser.add_argument('--base_log_dir', type=str, default='./log')
    parser.add_argument('--use_next_obs_in_context', type=bool, default=False)
    parser.add_argument('--recurrent', type=bool, default=True)
    parser.add_argument('--train_steps', type=int, default=50)
    parser.add_argument('--collect_data_samples', type=int, default=50)
    parser.add_argument('--random_steps', type=int, default=0)  # 10000
    parser.add_argument('--update_begin_steps_rl', type=int, default=100000)  # 1000
    parser.add_argument('--update_begin_steps_latent', type=int, default=0)  # 1000

    parser.add_argument('--kl_lambda', type=float, default=0) #TODO 0.1
    parser.add_argument('--cpl_lambda', type=float, default=1)

    parser.add_argument('--latent_lr', type=float, default=3e-4)  # 1e-6
    parser.add_argument('--sac_lr', type=float, default=3e-4)  # 1e-3)
    parser.add_argument('--env_id_str', type=list, default=[2, 1, 0] + list(range(3, 12)))
    parser.add_argument('--z_deterministic', type=bool, default=True) #TODO False
    parser.add_argument('--pi_deterministic', type=bool, default=False)
    parser.add_argument('--rl_batch_size', type=int, default=128)
    parser.add_argument('--latent_batch_size', type=int, default=128)#256)

    parser.add_argument('--latent_fq', type=int, default=1)  # 5
    parser.add_argument('--rl_fq', type=int, default=1)  # 5
    parser.add_argument('-latent_ue', '--latent_encoder_update_every', type=int, default=5000)  # 5000
    parser.add_argument('-rl_ue', '--rl_update_every', type=int, default=50)  # 1e4
    parser.add_argument('-latent_bs', '--latent_buffer_size', type=int, default=1000000)  # 50000
    parser.add_argument('-rl_bs', '--rl_buffer_size', type=int, default=1000000)  # 5000

    parser.add_argument('--seq_len', type=int, default=20)  # 20
    parser.add_argument('--latent_dim', type=int, default=50)  # TODO 5
    parser.add_argument('--cuda_id', type=int, default=0)  # TODO
    parser.add_argument('--n_steps', type=float, default=3e6)  # TODO

    args = parser.parse_args()
    args.model_path='D:\study\code\\backup\LC-SAC\src\\result_cplsar_noQLoss_train_z_det_cbatch128\model_cpl_v3\Humanoid-v2_s0_l20_d50\\1_1_5000_50_1000000_1000000'

    for j in [0, 1, 2]:#, 3, 4]:
        args.seed = j
        torch.manual_seed(j)
        args.use_next_obs_in_context = False

        args.epochs = int(args.n_steps / args.steps_per_epoch)
        print(f'epochs:{args.epochs}')

        print('-' * 10)
        print(f'experiment {args.env} seed {j} begin!')
        print('-' * 10)
        run(args)
        print('-' * 10)
        print(f'experiment {args.env} seed {j} done!')
        print('-' * 10)
    print('-' * 10)
    print(f'experiment {args.env} all seed done!')
    print('-' * 10)
