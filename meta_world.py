import metaworld
import random
import time
import numpy as np

# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
env_id = 'reach-v1'  # 'pick-place-v1'
ml1 = metaworld.ML1(env_id)  # Construct the benchmark, sampling tasks

env = ml1.train_classes[env_id]()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks[:40])
env.set_task(task)  # Set task

# obs = env.reset()  # Reset environment
# a = env.action_space.sample()  # Sample an action
# obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action


# print(env.observation_space, env.action_space)
# print(env.observation_space.shape, env.action_space.shape)
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
    # time.sleep(0.1)
    path_length += 1
    rewards += r
    # print(a, r, d, env_info)
    o = next_o

    if d or path_length>=201:
        # time.sleep(0.2)
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
