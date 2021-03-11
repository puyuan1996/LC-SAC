import torch
from torch import nn as nn
from torch.nn import functional as F

class MLPLatentEncoder(nn.Module):
    '''
    encode context via recurrent network
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim,init_w=3e-3,device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = latent_dim * 2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_size)
        self.fc3.weight.data.uniform_(-init_w, init_w).to(device)
        self.fc3.bias.data.uniform_(-init_w, init_w).to(device)

    def forward(self, x):
        # batch_sz, seq, feat = x.size()
        # out = x.view(batch_sz, seq, feat)

        batch_sz, feat = x.size()
        out = x.view(batch_sz, feat)

        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(x))
        # out = out.view(batch_sz, seq, -1)

        # output layer
        out = F.relu(self.fc2(out))
        output = self.fc3(out)
        # output = F.relu(self.fc3(out))

        return output

class RecurrentLatentEncoder(nn.Module):
    '''
    encode context via recurrent network
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim,init_w=3e-3,device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = latent_dim * 2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_size)

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.hidden_in = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).to(device),
                          torch.zeros([1, 1, self.hidden_dim],
                                      dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

    def forward(self, x, hidden_in):
        batch_sz, seq, feat = x.size()
        out = x.view(batch_sz, seq, feat)

        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(x))
        out = out.view(batch_sz, seq, -1)

        out, lstm_hidden = self.lstm(out, hidden_in)
        # self.hidden = lstm_hidden
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        out = F.relu(self.fc2(out))
        output = self.fc3(out)
        # output = F.relu(self.fc3(out))

        return output, lstm_hidden

    # def reset(self, num_tasks=1):
    #     self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class RecurrentLatentEncoder2head(nn.Module):
    '''
    encode context via recurrent network
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim,init_w=3e-3,device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = latent_dim * 2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean_linear = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_std_linear = nn.Linear(self.hidden_dim, self.latent_dim)

        self.mean_linear.weight.data.uniform_(-init_w, init_w).to(device)
        self.mean_linear.bias.data.uniform_(-init_w, init_w).to(device)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w).to(device)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w).to(device)

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.hidden_in = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).to(device),
                          torch.zeros([1, 1, self.hidden_dim],
                                      dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

    def forward(self, x, hidden_in):
        batch_sz, seq, feat = x.size()
        # out = x.view(batch_sz, seq, feat)

        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(x))
        out = out.view(batch_sz, seq, -1)

        out, lstm_hidden = self.lstm(out, hidden_in)
        # (-inf,+inf)

        if seq == 1:
            # take the last hidden state to predict z
            out = out[:, -1, :]
        else:
            # 训练时输出z_t和z_t+1
            out = out[:, -2:, :]

        # output layer
        out = F.relu(self.fc2(out))
        # output = self.fc3(out)
        mu = self.mean_linear(out)
        # mu = torch.tanh(mu)

        log_std = self.log_std_linear(out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # very important

        output = torch.cat((mu, log_std), dim=1)
        # output = torch.tanh(output)

        # with torch.no_grad():
        #     output.fill_(0)

        return output, lstm_hidden

class RecurrentLatentEncoder2head_v4(nn.Module):
    '''
    encode context via recurrent network
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim,init_w=3e-3,device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = latent_dim * 2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean_linear = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_std_linear = nn.Linear(self.hidden_dim, self.latent_dim)

        self.mean_linear.weight.data.uniform_(-init_w, init_w).to(device)
        self.mean_linear.bias.data.uniform_(-init_w, init_w).to(device)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w).to(device)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w).to(device)

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.hidden_in = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).to(device),
                          torch.zeros([1, 1, self.hidden_dim],
                                      dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

    def forward(self, x, hidden_in):
        batch_sz, seq, feat = x.size()
        # out = x.view(batch_sz, seq, feat)

        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(x))
        out = out.view(batch_sz, seq, -1)

        out, lstm_hidden = self.lstm(out, hidden_in)
        # (-inf,+inf)

        if seq == 1:
            # take the last hidden state to predict z
            out = out[:, -1, :]
        # else:
        #     # 训练时输出z_t和z_t+1
        #     # out = out[:, -2:, :]
        #     out = out[:, :, :]
        # output layer
        out = F.relu(self.fc2(out))
        # output = self.fc3(out)
        mu = self.mean_linear(out)
        # mu = torch.tanh(mu)

        log_std = self.log_std_linear(out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)  # very important

        output = torch.cat((mu, log_std), dim=-1)
        # output = torch.stack([mu, log_std], dim=2)
        # output = torch.tanh(output)

        return output, lstm_hidden

class RecurrentLatentEncoderDet(nn.Module):
    '''
    encode context via recurrent network
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim,init_w=3e-3,device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_size = latent_dim * 2
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        # self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean_linear = nn.Linear(self.hidden_dim, self.latent_dim)
        # self.log_std_linear = nn.Linear(self.hidden_dim, self.latent_dim)

        self.mean_linear.weight.data.uniform_(-init_w, init_w).to(device)
        self.mean_linear.bias.data.uniform_(-init_w, init_w).to(device)
        # self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.hidden_in = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float).to(device),
                          torch.zeros([1, 1, self.hidden_dim],
                                      dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

    def forward(self, x, hidden_in):
        batch_sz, seq, feat = x.size()
        # out = x.view(batch_sz, seq, feat)

        out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(x))
        out = out.view(batch_sz, seq, -1)

        out, lstm_hidden = self.lstm(out, hidden_in)
        # (-inf,+inf)

        if seq == 1:
            # take the last hidden state to predict z
            out = out[:, -1, :]
        else:
            # 训练时输出z_t和z_t+1
            out = out[:, -2:, :]

        # output layer
        out = F.relu(self.fc2(out))
        # output = self.fc3(out)
        mu = self.mean_linear(out)
        # mu = torch.tanh(mu) #TODO

        return mu, lstm_hidden