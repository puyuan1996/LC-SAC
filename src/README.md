### Latent Context Based Soft Actor-Critic 
Experements for LC-SAC-Seq

### 主要
- LC-SAC-Seq主要实现位于 /spinup/algos/pytorch/sac/
- latent encoder 基于LSTM实现位于latent_encoder.py
- test_model.py 测试训练得到的model


### 网络结构
q, pi networks (256,256)
latent_encoder (128,lstm,128)

### 测试实验 
main_sac.py
agent_sac.py
sac.py
把z置为0，hidden_dim=1, latent_encoder hidden_size 1
即测试原sac算法是否可行，实验证明是可以跑出spinup sac的结果

 