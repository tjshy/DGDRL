# Dynamic Graph-based Deep Reinforcement Learning with Long and Short-term Relation Modeling for Portfolio Optimization (DGDRL)

##1. Dataset:
   The 'data' directory contains the data for our three real-world datasets. The csv files contain historical feature data, 'relation.pkl' stores the raw graphs for short-term relations, and 'relation2.pkl' holds the final graphs for short-term relations after threshold filtering. The npy files store the graph for long-term relations.

##2. Train:
   Set the parameters in 'net.py', 'main.py' and 'stock\_env.py'. And run the 'main.sh' to train the model. The parameters are as follows:

``` config
 num_stocks = 30             # the number of stocks
 window_len = 20             # the length of look-back window
 hidden_dim = 128            # hidden_dim
 output_dim = 128            # output_dim
 dropout = 0.3               # dropout
 negative_slope=0.2          # negative_slope
 num_heads = 8               # the number of attention heads
 random_seed=42              # seed
 action_dim = 30             # action space(equal to the number of stocks)
 epoch = 200                 # the number of training epochs
 beg_idx = 0                 # the start position of the training set
 end_idx = 3163              # the end position of the training set
 max_step = 3404             # the end position of the entire set
 beg_idx = 3143(3163-20)     # the start position of the test set(The test set follows the training set, and the first 20 days are empty.)
 end_idx = 3404              # the end position of the test set
 gamma=0.99                  # profit rate after deducting stock transaction fees
 ```



