result_path='/home/DGDRL/'
class Config:
    def __init__(self) -> None:
        '''
        1. data source
        '''
        self.dataset = 'SSE50'
        self.npz_pwd = '/home/DGDRL/data'+self.dataset+'/'+self.dataset+'.numpy.npz'
        self.csv_pwd = '/home/DGDRL/data'+self.dataset+'/'+self.dataset+'.csv'
        self.relation_pwd = '/home/DGDRL/data/'+self.dataset+'/'+'relation_2.pkl'
        if self.dataset == 'SSE50':
            self.featureList = ["open_r","high_r","low_r","close_r","turnover_r","volume_r","macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
        else :
            self.featureList = ["open_r","high_r","low_r","close_r","volume_r","macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
        '''
        2. train and test
        '''
        self.gpu_id = 1
        self.random_seed=40
        self.thread_num = 16
        self.window=20
        self.epoch=300
        self.env_name = "StockTradingEnv-v2" 

        '''
        3. network parameter
        '''
        self.num_stocks = 30
        self.input_dim=len(self.featureList)
        self.hidden_dim = 128
        self.output_dim = 128
        self.dropout = 0.3
        self.num_heads=8
        self.negative_slope=0.2
        self.gamma = 0.99 
        self.initial_amount=1e6 
        self.batch_size = 128
        self.repeat_times = 4
        self.reward_scale = 1
        self.learning_rate_cri = 0.08
        self.learning_rate_act = 0.08
        self.ratio_clip = 0.25
        self.lambda_entropy = 0.02
        self.train_beg_idx = '2008-01-03'
        self.train_end_idx = '2020-12-03'
        self.test_beg_idx = '2020-12-04'
        self.test_end_idx='2021-12-30'   
        self.action_dim = 30
        self.state_input_dim = 128 
        self.act_mid_layer_num=2
        self.act_mid_dim = 128
        self.cri_mid_layer_num=2
        self.cri_mid_dim = 128
        