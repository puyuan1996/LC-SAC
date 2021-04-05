注意：该代码库中的代码与产生原论文实验结果的代码不同，该代码库中latent context encoder为LSTM结构
Our code is based on https://spinningup.openai.com and https://github.com/katerakelly/oyster
### Latent Context Based Soft Actor-Critic

### Note
参见sac_q_no_delay.py 最小化q_loss
- s,a,r,s' 在q loss中target r_t + gamma*(q_min(z_t+1, s_t+1, a^_t+1) - alpha*log_pi(a^_t+1 | z_t+1,s_t+1)) 中的latent context 为z_t+1
q(z_t,s_t,a_t)中的为z_t 不能使z_t+1 简单取为z_t
- 下面的技巧似乎非常重要
latent_encoder.py
LOG_STD_MAX = 2
LOG_STD_MIN = -20
log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

### latent encoder
latent_encoder为lstm网络,每步输入hidden_in, (s,a,r) 输出 hidden_out, c
c为10维向量，表示context的均值与方差，然后从中采样得到 5维的向量 c, 期望其表征环境的dynamics和agent过去的behaviour

借鉴VAE latent_encoder_loss 为最小化c与均值为0，方差为1的高斯分布的同时，最小化q_loss (pi_loss)
期望学到的latent context为有利于学习好的策略的表征，而不仅仅是像VAE一样重建原状态
loss_context = cpl+a*ql+b*kl_loss

### q1,q2,pi设计与SAC原始算法一致
只是输入把z 与原始状态观测值o 拼接在一起


### Reference
[1] https://github.com/katerakelly/oyster
[2] K. Rakelly, A. Zhou, D. Quillen, C. Finn, S. Levine. ”Efficient off-policy Meta-reinforcement learning via probabilistic context variables,” in Proceedings of International Conference on Machine Learning (ICML), 2019.

