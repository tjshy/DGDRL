
result_path='/home/NDX/test/'
class Config:
    def __init__(self) -> None:
        '''
        1. data
        '''
        self.dataset = 'NDX'
        self.npz_pwd = '/home/'+self.dataset+'/'+self.dataset+'.numpy.npz'
        self.csv_pwd = '/home/'+self.dataset+'/'+self.dataset+'.csv'
        self.relation_pwd = '/home/'+self.dataset+'/'+'d_relation.pkl'

        self.featureList = ['close','open','high','low','volume']
        '''
        2. train and test
        '''
        self.gpu_id = 1
        self.random_seed=40
        self.thread_num = 16
        self.window=20
        self.epoch=800
        self.env_name = "StockTradingEnv-v2" 
        '''
        3. parameters
        '''
        self.num_stocks = 96
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
        self.train_beg_idx = '2020-01-02'
        self.train_end_idx = '2022-12-01'
        self.test_beg_idx = '2022-12-01'
        self.test_end_idx='2023-12-29'   
        self.state_input_dim = 128 
        self.action_dim = self.num_stocks
        self.act_mid_layer_num=2
        self.act_mid_dim = 128
        self.cri_mid_layer_num=2
        self.cri_mid_dim = 128