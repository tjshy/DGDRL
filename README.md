# Dynamic Graph-based Deep Reinforcement Learning with Long and Short-term Relation Modeling for Portfolio Optimization (DGDRL)

##1. Dataset:
   The 'data' directory contains the data for our three real-world datasets. The csv files contain historical feature data, 'relation.pkl' stores the raw graphs for short-term relations, and 'relation2.pkl' holds the final graphs for short-term relations after threshold filtering. The npy files store the graph for long-term relations. It is important to note that training is conducted on a daily basis, with the division between training and testing set on a daily basis as well.

##2. Train:
   Set the parameters in 'config.py'. And run the 'main.sh' to train the model. The parameters are as follows:

``` config
 gpu_id = 1                  # gpu_id
 random_seed = 40            # seed
 thread_num = 16             # thread_number
 window = 20                 # the length of look-back window
 epoch = 300                 # the number of training epochs
 num_stocks = 30             # the number of stocks
 hidden_dim = 128            # hidden_dim
 output_dim = 128            # output_dim
 dropout = 0.3               # dropout
 num_heads = 8               # the number of attention heads
 negative_slope = 0.2        # negative_slope
 gamma = 0.99                # gamma
 initial_amount = 1e6        # initial funds for stock trading
 batch_size = 128            # batchsize
 repeat_times = 4            # the number of times each epoch updates the network
 reward_scale = 1            # rewardscale
 learning_rate_cri = 0.08    # learning rate of the critic
 learning_rate_act = 0.08    # learning rate of the agent
 ratio_clip = 0.25           # clipping ratio
 lambda_entropy = 0.02       # lambda entropy
 train_beg_idx = '2008-01-03'# the start date of training
 train_end_idx = '2020-12-03'# the end date of training (set aside 20 days as a window for testing)
 test_beg_idx = '2020-12-04' # the start date of test
 test_end_idx = '2021-12-30' # the end date of test
 action_dim = 30             # action space(equal to the number of stocks)
 state_input_dim = 128       # equal to hidden_dim
 act_mid_layer_num=2         # number of network layers for actor
 act_mid_dim = 128           # network dimension of actor
 cri_mid_layer_num=2         # number of network layers for critic
 cri_mid_dim = 128           # network dimension of critic
 ```



