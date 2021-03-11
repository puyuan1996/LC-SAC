import torch.nn as nn
import torch
import torch.nn.functional as F


class contrast(nn.Module):
    def __init__(self, act=True, z_dim=5, obs_dim=11, encoder=None, encoder_targ=None,device=None):
        super(contrast, self).__init__()

        self.encoder = encoder
        self.encoder_targ = encoder_targ
        self.device = device
        if act:
            self.W = nn.Sequential(
                nn.Linear(z_dim * 2 + action_dim, 256), nn.ReLU(),
                nn.Linear(256, z_dim * 2), nn.LayerNorm(z_dim * 2))
            # self.W = nn.Sequential(
            #     nn.Linear(z_dim, 256), nn.ReLU(),
            #     nn.Linear(256, z_dim), nn.LayerNorm(z_dim))
        else:
            self.W = nn.Parameter(torch.rand(z_dim, obs_dim)).to(self.device) # requires_grad=True

    def encode(self, x, detach=False, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.encoder_targ(x)
        else:
            z_out = self.encoder(x)
        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_anchor, z_pos, a=None):
        if a is not None:
            assert z_anchor.size(0) == a.size(0)
            Wz = self.W(torch.cat([z_anchor, a], dim=1))  #  (B,z_dim)
            #  Wz = self.W(z_a) # (B,z_dim)
            logits = torch.matmul(Wz, z_pos.T)
        else:
            Wz = torch.matmul(self.W, z_pos.T)  #  (z_dim/z_dim+a_dim,B)
            logits = torch.matmul(z_anchor, Wz)  #  (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]

        return logits


    def loss_tcl(self, z_anchor,data):
        z_pos = data['obs'].to(self.device)

        # z_a, mu_a, std_a = self.clrl_t.encode(o_anchor)
        # h_a = torch.cat([mu_a, std_a], dim=1)

        # z_pos, mu_pos, std_pos = self.contrast.encode(o_pos, ema=True)
        # h_pos = torch.cat([mu_pos, std_pos], dim=1)

        #  time contrastive loss
        logits = self.compute_logits(z_anchor, z_pos)
        labels = torch.arange(logits.shape[0]).long()#.to(self.device)
        loss_tcl = F.cross_entropy(logits, labels)
        """InfoNCE loss"""
        # #  vae loss
        # rec_obs = self.clrl_t.encoder.decode(z_a)
        # loss_vae = F.mse_loss(rec_obs, o_anchor)

        return loss_tcl
