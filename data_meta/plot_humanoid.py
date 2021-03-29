import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

env_id_dict = {0: 'Ant-v2', 1: 'HalfCheetah-v2', 2: 'Hopper-v2', 3: 'Humanoid-v2', 4: 'Pusher-v2', 5: 'Reacher-v2',
               6: 'Striker-v2', 7: 'Swimmer-v2', 8: 'Thrower-v2'}
learner_path_dict = {0: 'ant', 1: 'halfcheetah', 2: 'hopper', 3: 'humanoid', 4: 'pusher', 5: 'reacher', 6: 'striker',
                     7: 'swimmer', 8: 'thrower'}
env_id = env_id_dict[3]

df0 = pd.read_csv("v4_Humanoid-v2_l8_5_1_5000_50_1000000_1000000_test_rew/Humanoid-v2_s1_l8_5_1_5000_50_1000000_1000000_test_rew_0.csv")
df1 = pd.read_csv("v4_Humanoid-v2_l8_5_1_5000_50_1000000_1000000_test_rew/Humanoid-v2_s1_l8_5_1_5000_50_1000000_1000000_test_rew_1.csv")
df2 = pd.read_csv("v4_Humanoid-v2_l8_5_1_5000_50_1000000_1000000_test_rew/Humanoid-v2_s1_l8_5_1_5000_50_1000000_1000000_test_rew_2.csv")
sac0 = pd.read_table('../sac_results/sac_Humanoid-v2/sac_Humanoid-v2_s0/progress.txt', sep='\t')

# sac=sac0[:int(3e6/4000)]
sac=sac0[:int(2.5e6/4000)]
data = [df0[:int(3e6/10000)],df1[:int(3e6/10000)],df2[:int(3e6/10000)]]
if isinstance(data, list):
    data = pd.concat(data, ignore_index=True)

# sac = [sac1[:int(2e6) //4000],sac2[:int(2e6)//4000]]
# if isinstance(sac, list):
#     sac = pd.concat(sac, ignore_index=True)


sns.set(style="darkgrid", font_scale=1.5)
# ax0 = sns.lineplot(x='Number of train steps total', y='total_rewards_mean', hue=None,data=data, err_style='band')#,legend='LC-SAC')#, **kwargs)
ax0 = sns.lineplot(x='Step', y='Value', hue=None, data=data, ci='sd', err_style='band')
ax1 = sns.lineplot(x='TotalEnvInteracts', y='AverageEpRet', hue=None, data=sac, ci='sd', err_style='band')
# ax1= sns.lineplot(x='TotalEnvInteracts', y='AverageEpRet', hue=None,data=df[:int(2e6)//4000], err_style='band')#,legend='SAC')#, **kwargs)
# handles, labels = ax0.get_legend_handles_labels()
# plt.plot(x, y, '-', label='LC-SAC', color='g')
plt.xlabel('Training Steps')
plt.ylabel('Performance')
plt.title(f'{env_id}')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend(loc='upper left',labels=['LC-SAC', 'SAC'])
plt.show()