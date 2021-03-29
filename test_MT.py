import metaworld
import random
import time
import numpy as np

# print(metaworld.MT1.ENV_NAMES)  # Check out the available environments
# print(metaworld.MT10)  # Check out the available environments

ml10 = metaworld.ML10()  # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
    env = env_cls()
    task = random.choice([task for task in ml10.train_tasks
                          if task.env_name == name])
    env.set_task(task)
    training_envs.append(env)

# for env in training_envs:
#     # print(env)
#     # print(env.observation_space, env.action_space)
#     # print(env.observation_space.shape, env.action_space.shape)
#     obs = env.reset()  # Reset environment
#     print(obs)
#     a = env.action_space.sample()  # Sample an action
#     obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action

env=random.choice(training_envs)
test_eps = 1
rewards = 0
path_length = 0
ep = 0
total_ave_rewards = 0
total_rewards = []
o = env.reset()

while ep < test_eps:
    a = env.action_space.sample()
    next_o, r, d, env_info = env.step(a)
    # img=env.render(mode='rgb_array')
    # print(img.shape) #(400, 600, 3)
    # plt.savefig('img.png',img)
    env.render()
    # time.sleep(10)
    path_length += 1
    rewards += r
    # print(a, r, d, env_info)
    o = next_o

    if d or path_length >= 201:
        time.sleep(10)
        print(f'ep:{ep},rewards:{rewards},path_length:{path_length},')  # ,ave rewards:{rewards / path_length}
        total_rewards.append(rewards)
        rewards = 0
        path_length = 0
        ep += 1

        # Set seeds
        seed = ep
        # torch.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)
        o = env.reset()
env.close()
total_rewards = np.asarray(total_rewards)
print(
    f'total_rewards\n mean: {total_rewards.mean()},std: {total_rewards.std()},max: {total_rewards.max()},min: {total_rewards.min()}')
