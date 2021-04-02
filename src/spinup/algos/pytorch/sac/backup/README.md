##sac_q.py 
latent_encoder 优化q loss 
使z_t+1 简单取为z_t

##sac_q_z2.py 
latent_encoder 优化q loss 
没有使z_t+1 简单取为z_t，而是单独计算

### sac.py
lssac 
把z置为0，hidden_dim=1, latent_encoder hidden_size 1
即测试原sac算法是否可行，实验证明是可以跑出spinup sac的结果

### sac_q_v0.py
用collect_data()函数

### sac_q_v1.py
仿照spinup sac.py
不用collect_data()函数

### sac_q_v2.py
q,pi 和latent encoder交替训练
不保存z

### sac_q_v3.py
q,pi 和latent encoder交替训练
训练latent encoder时loss为q loss和kld
rl训练时，将z存入rl replay buffer,
Ant-v2 HalfCheetah-v2 只能在这个条件下跑出结果

### sac_q_v4.py
q,pi 和latent encoder交替训练
训练latent encoder时loss为q loss和kld
rl训练时，将z存入rl replay buffer,s
采样得到长度为8的sequence,传入encoder,得到的8个z，得到4步的q_loss,相加后梯度回传更新encoder参数

### sac_q_tcl_v3.py
可能有错误？
q,pi 和latent encoder交替训练
rl训练时，将z存入rl replay buffer,
训练latent encoder时loss为q loss, tcl loss和kld
Hopper-v2不稳定
