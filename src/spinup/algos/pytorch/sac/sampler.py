import numpy as np
import torch

def rollout(env, agent, max_path_length=np.inf, deterministic=False, render=False,random_action=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    o = env.reset()
    path_length = 0
    agent.clear_z()  # 从先验中采样
    hidden_in = (torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(agent.device),
                 torch.zeros([1, 1, agent.hidden_dim], dtype=torch.float).to(agent.device))
    while path_length < max_path_length:
        if random_action:
            a = env.action_space.sample()
        else:
            if path_length ==0:
                # a, hidden_out = agent.get_action(hidden_in, o, deterministic=deterministic)
                hidden_out = hidden_in
                a = env.action_space.sample()
            else:
                a, hidden_out = agent.get_action(hidden_in, o, deterministic=deterministic)
            hidden_in = hidden_out
        a = a.squeeze()
        next_o, r, d, _ = env.step(a)
        agent.update_context([o, a, r, next_o, d])

        path_length += 1
        if render:
            env.render()
        observations.append(o)
        actions.append(a)
        rewards.append(r)
        terminals.append(d)
        o = next_o

        # agent.infer_posterior(agent.context[:, -100:, :])
        # if path_length > 1:
        #     agent.infer_posterior(agent.context[-20:, :].unsqueeze(0))  # 包含agent.sample_z()
        # else:
        #     agent.sample_z()  # 每一步从先验中采样

        if d:
            break

    actions = np.stack(actions)
    observations = np.stack(observations)
    next_observations = np.stack(observations[1:] + [o])

    return dict(
        obs=observations,  # (path_length, obs_dim)
        act=actions,  # (path_length, act_dim)
        rew=np.array(rewards).reshape(-1, 1),  # (path_length, 1)
        obs2=next_observations,
        done=np.array(terminals).reshape(-1, 1), # (path_length, 1)
    )


def obtain_rollout_samples(env, agent, max_path_length, max_samples, max_nums_paths=np.inf, deterministic=False,random_action=False):
    """
    Obtains samples in the environment until either we reach either max_samples transitions or
    num_traj trajectories.
    The resample argument specifies how often (in trajectories) the agent will resample it's context.
    """

    assert max_samples < np.inf or max_nums_paths < np.inf, "either max_samples or max_trajs must be finite"

    paths = []
    total_steps = 0
    nums_paths = 0
    while total_steps < max_samples and nums_paths < max_nums_paths:
        path = rollout(env, agent, max_path_length=max_path_length, deterministic=deterministic,
                     render=False,random_action=random_action)
        paths.append(path)
        total_steps += len(path['obs'])
        nums_paths += 1
    return paths, total_steps
