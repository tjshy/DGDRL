###Transformer
```
hidden_size: 64
num_layers: 2
num_heads: 4
dropout: 0.1
lr: 0.0002
n_epochs: 1000
max_steps_per_epoch: 100
early_stop: 10
batch_size: 1024
lamb: 1.0
rho: 0.99
```
###TRA
```
hidden_size: 64
num_layers: 2
num_heads: 4
dropout: 0.1
lr: 0.0005
n_epochs: 500
max_steps_per_epoch: 100
early_stop: 20
num_states 3
batch_size: 512
lamb: 1.0
rho: 0.99
```
The parameters of transformer and TRA are mainly referred to https://github.com/microsoft/qlib/blob/main/examples/benchmarks/TRA/configs.

###CGM
```
num_layers: 1 
max_text_len: 50
max_sentence: 10
epoch: 30 
encoder: 'rnn'
graph_encoder: 'lstm'
bidirec: True
edge_threshold: 0.6
label_size: 1
vocab_size: 0
feature_size: 8
relation_num: 2
param_init: 0.1
optim: 'adam'
learning_rate: 0.0001
filter_size: 100
learning_rate_decay: 0.95
hidden_size: 300
bert_size: 768
dropout: 0.0
```
The parameters of CGM are mainly referred to https://github.com/lancopku/CGM/blob/main/config_regression.yaml.

###THGNN
```
adj_threshold = 0.1        
max_epochs = 60             
epochs_eval = 10            
epochs_save_by = 60         
lr = 0.0002                
gamma = 0.3                 
hidden_dim = 128          
num_heads = 8               
out_features = 32           
dropout = 0.1             
batch_size = 1              
loss_fcn = mse_loss   
epochs_save_by = 60 
```
The parameters of THGNN are mainly referred to https://github.com/finint/THGNN.
###FactorVAE
```
batch_size = 32
latent_size = 64
factor_size = 32
time_span = 60
gru_input_size = 64
hidden_size = 64
lr = 1e-5
epochs = 100
```
The parameters of FactorVAE are mainly referred to https://github.com/ytliu74/FactorVAE.
###CTTS
```
hidden_size: 64
num_layers: 2
num_heads: 4
dropout: 0.1
lr: 0.0002
n_epochs: 1000
early_stop: 10
lamb: 1.0
rho: 0.99
batch_size: 512
```
There is no open source code for this method, we use the framework implementation of qlib, and the parameters are similar to the Transformer.

###PPO

```
hidden_dim = 128
output_dim = 128
dropout = 0.3
num_heads=8
negative_slope=0.2
gamma = 0.99
initial_amount=1e6
batch_size = 128
repeat_times = 4
reward_scale = 1
learning_rate_cri = 0.08
learning_rate_act = 0.08
ratio_clip = 0.25
lambda_entropy = 0.02
```
The parameter Settings of PPO are similar to those of our model.

###AlphaStock

```
gamma = 0.99
k_epochs = 500
actor_lr = 0.003 
critic_lr = 0.003 
eps_clip = 0.2
entropy_coef = 0.01 
update_freq = 16 
actor_hidden_dim = 64
critic_hidden_dim = 64
n_encoder_layers = 1 
n_heads = 8 
negative_slope = 0.2
```
There is no open source code for this method, we reproduce it and adjust the parameters.

###DeepTrader
```
lr: 1e-06,
hidden_dim: 128,
num_blocks: 4,
kernel_size: 2,
dropout: 0.5,
window_len: 13,
max_steps: 12,
gamma: 0.05,
tau: 0.001,
batch_size: 37,
trade_mode: "D",
fee: 0.001,
num_rounds: 80000,
norm_type: "standard",
epochs: 1500,
trade_len: 21
```
The parameters of DeepTrader are mainly referred to https://github.com/CMACH508/DeepTrader/blob/main/src/hyper.json.

###DeepPocket
```
gamma=0.99,
num_episodes=1000   
batch_size=32,
actor_lr=1e-3,
critic_lr=1e-4,
actor_weight_decay=1e-8,
critic_weight_decay=1e-8,
cheb_k=3,
gnn_input_channels= 3
gnn_hidden_channels='8,8,8'
gnn_output_channels= 3, help
mem_size= 200
sample_bias= 1e-5
```
The parameters of DeepPocket are mainly referred to https://github.com/MCCCSunny/DeepPocket.