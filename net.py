

import torch
import torch.nn as nn
import numpy as np
from config import Config
cfg = Config()
gpu_id = cfg.gpu_id
num_stocks = cfg.num_stocks
input_dim = cfg.input_dim
hidden_dim = cfg.hidden_dim
output_dim = cfg.output_dim
dropout = cfg.dropout
num_heads = cfg.num_heads
negative_slope = cfg.negative_slope
window_len = cfg.window
device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
SGRAPH_SSE = torch.as_tensor(np.load("DGDRL/data/SSE50/SGRAPH_SSE.npy"), dtype=torch.float32).to(device)
SGRAPH_DOW = torch.as_tensor(np.load("DGDRL/data/DJI30/SGRAPH_DJI.npy"), dtype=torch.float32).to(device)
SGRAPH_NAS = torch.as_tensor(np.load("DGDRL/data/NAS100/SGRAPH_NAS.npy"), dtype=torch.float32).to(device)


def build_fcn(mid_dim, mid_layer_num, inp_dim, out_dim):  # fcn (Fully Connected Network)
    net_list = [nn.Linear(inp_dim, mid_dim), nn.ReLU(), ]
    for _ in range(mid_layer_num):
        net_list += [nn.Linear(mid_dim, mid_dim), nn.ReLU(), ]
    net_list += [nn.Linear(mid_dim, out_dim), ]
    return nn.Sequential(*net_list)


class DotProductAttention(nn.Module):
    def __init__(self, dropout = dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1,2)) / np.sqrt(d)
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(attention_weights), values)

class LSTM_HA(nn.Module):
    '''
    Here we employ the attention to LSTM to capture the time series traits more efficiently.
    '''
    def __init__(self, in_features,
            num_stocks = num_stocks,
            window_len = window_len,
            hidden_dim = hidden_dim,
            output_dim = output_dim,):
        super(LSTM_HA, self).__init__()
        self.in_features = in_features
        self.num_stocks = num_stocks
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=output_dim,batch_first=True)
        self.attention = DotProductAttention()


    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        h = self.attention(h_n, h_n, h_n)
        return h[0]
class GAT_MultiHeads(nn.Module):
    '''
    Here we employ the multi-channel graph attention mechanism.
    '''
    def __init__(self, in_features, 
            out_features=output_dim, 
            negative_slope=negative_slope, 
            num_heads=num_heads, 
            bias=True, residual=True):
        super(GAT_MultiHeads, self).__init__()
        self.num_heads = num_heads
        self.out_features = int(out_features / self.num_heads)
        self.weight = nn.Linear(in_features, self.num_heads * self.out_features)
        self.weight_u = nn.Parameter(torch.FloatTensor(self.num_heads, self.out_features, 1))
        self.weight_v = nn.Parameter(torch.FloatTensor(self.num_heads, self.out_features, 1))
        self.weight_cat = nn.Linear(self.num_heads * self.out_features * 2,self.num_heads * self.out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, self.num_heads*self.out_features)
        else:
            self.project = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, self.num_heads * self.out_features))
        else:
            self.register_parameter('bias', None)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight_u.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight_v.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.bias.data, gain=nn.init.calculate_gain('relu'))
    def forward(self, inputs, Dgraph=None, Sgraph=SGRAPH_SSE, requires_weight=False):

        stockNum = inputs.shape[0]
        score = self.weight(inputs)
        score = score.reshape(stockNum, self.num_heads, self.out_features).permute(dims=(1, 0, 2))
        f_1 = torch.matmul(score, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(score, self.weight_v).reshape(self.num_heads, -1, 1)
        logits = f_1 + f_2
        weight = self.leaky_relu(logits)
        Dgraph = Dgraph.unsqueeze(0).repeat((self.num_heads, 1, 1))
        D_masked_weight = torch.mul(weight, Dgraph).to_sparse()
        D_attn_weights = torch.sparse.softmax(D_masked_weight, dim=2).to_dense()
        D_support = torch.matmul(D_attn_weights, score)
        D_support = D_support.permute(dims=(1, 0, 2)).reshape(stockNum, self.num_heads * self.out_features)
        Sgraph = Sgraph.unsqueeze(0).repeat((self.num_heads, 1, 1))
        S_masked_weight = torch.mul(weight, Sgraph).to_sparse()
        S_attn_weights = torch.sparse.softmax(S_masked_weight, dim=2).to_dense()
        S_support = torch.matmul(S_attn_weights, score)
        S_support = S_support.permute(dims=(1, 0, 2)).reshape(stockNum, self.num_heads * self.out_features)

        score = torch.cat([S_support,D_support], dim=1)
        score = self.weight_cat(score)
        if self.bias is not None:
            score = score + self.bias
        if self.residual:
            score = score + self.project(inputs)
        if requires_weight:
            return score, D_attn_weights, S_attn_weights
        else:
            return score
class FeatureHead(nn.Module):
    '''
    Here is the whole perception layer
    '''
    def __init__(self, 
            input_dim=input_dim, 
            output_dim = output_dim, 
            num_stocks=num_stocks, 
            window=window_len, 
            hidden_dim = hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_stocks = num_stocks
        self.window = window
        self.lstm = LSTM_HA(
                        in_features=self.input_dim
                        ,num_stocks=self.num_stocks
                        ,output_dim=self.hidden_dim
                        )
        self.gat = GAT_MultiHeads(
                        in_features=self.hidden_dim
                        ,out_features=self.output_dim
                        )
        self.gat.reset_parameters()
        
    def forward(self, x, x_dg=None):
        scores = self.lstm(x)
        scores = self.gat(scores, Dgraph=x_dg)
        return scores

class Actor(nn.Module):
    def __init__(self, mid_dim, mid_layer_num, state_dim, action_dim):
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim, out_dim=1)
        self.head = FeatureHead()
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state, drelation=SGRAPH_SSE):
        action = self.net(self.head(state, drelation)).transpose(1,0).softmax(dim=1)
        return action

    def get_action(self, state, drelation, soft_noise = 0.5):
        a_avg = self.net(self.head(state, drelation)).transpose(1,0).softmax(dim=1)
        a_std = self.a_std_log.exp()
        noise = torch.randn_like(a_avg) * soft_noise
        action = (a_avg + noise * a_std).softmax(dim=1)
        return action, noise

    def get_logprob(self, state, action, drelation):
        a_avg = self.net(self.head(state, drelation)).transpose(1,0).softmax(dim=1)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)

        return log_prob

    def get_logprob_entropy(self, state, action, drelation):
        a_avg = self.net(self.head(state, drelation)).transpose(1,0).softmax(dim=1)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1) 

        dist_entropy = (logprob.exp() * logprob).mean()
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise): 
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1) 


class Critic(nn.Module):
    def __init__(self, mid_dim, mid_layer_num, state_dim, action_dim):
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim, out_dim=1)
        self.tail = nn.Linear(action_dim,1)
        self.head = FeatureHead()
    def forward(self, state, drelation=SGRAPH_SSE):
        score = self.net(self.head(state, drelation)).reshape(1,-1) 
        score = self.tail(score)
        return score
