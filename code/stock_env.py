# stock.env

import os
import numpy as np
import pandas as pd
import numpy.random as rd
from tqdm import *
class StockTradingEnv:
    def __init__(self,cfg,mode):#
        if mode == 'train':
            beg_idx=cfg.train_beg_idx
            end_idx=cfg.train_end_idx
        elif mode == 'test':
            beg_idx=cfg.test_beg_idx
            end_idx=cfg.test_end_idx
        self.source = cfg.dataset
        self.npz_pwd = cfg.npz_pwd
        self.csv_pwd = cfg.csv_pwd
        self.relation_pwd = cfg.relation_pwd
        self.featureList=cfg.featureList
        self.npz_pwd = cfg.npz_pwd
        self.csv_pwd = cfg.csv_pwd
        self.relation_pwd = cfg.relation_pwd
        
        self.close_ary, self.tech_ary, self.Drelation = self.load_data_from_disk(beg_idx,end_idx)

        print(f"| StockTradingEnv: close_ary.shape {self.close_ary.shape}")
        print(f"| StockTradingEnv: tech_ary.shape {self.tech_ary.shape}")
        print(f"| StockTradingEnv: Drelation.shape {self.Drelation.shape}")

        self.initial_amount = cfg.initial_amount
        self.gamma = cfg.gamma
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = 0
        self.everyday_asset=[]
        self.daily_return = []

        self.window=cfg.window
        self.shares_num = self.close_ary.shape[1]# 股票数

        # environment information
        self.env_name = cfg.env_name
        self.state_dim = (self.shares_num,self.window,30)
        self.action_dim = self.shares_num
        self.if_discrete = False
        self.max_step = len(self.close_ary)

    def reset(self):
        self.day = self.window
        self.rewards = [0.0]
        self.total_asset = self.initial_amount
        self.everyday_asset=[]
        self.everyday_asset.append(self.total_asset)
        self.daily_return=[1.0]
        state = self.get_state()
        return state

    def get_state(self):
        state = np.hstack(([self.tech_ary[i] for i in range(self.day-self.window,self.day)])).reshape(self.window,self.shares_num,-1).transpose(1,0,2)
        return state, self.Drelation[self.day - self.window]

    def step(self, action):
        self.day += 1
        state = self.get_state()
        reward = sum((action*self.total_asset)*self.close_ary[self.day]/self.close_ary[self.day-1]) - self.total_asset
        self.rewards.append(reward)
        self.total_asset += reward
        self.everyday_asset.append(self.total_asset)
        self.daily_return.append(self.everyday_asset[-1]/self.everyday_asset[-2])
        done = self.day == self.max_step - 1
        if done:
            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.cumulative_returns = self.total_asset / self.initial_amount
        return state, reward, done, {}

    def load_data_from_disk(self, begin_index,end_index):
        tech_id_list = self.featureList
        df = pd.read_csv(self.csv_pwd,index_col = "Unnamed: 0")
        beg_idx=np.argwhere(df.date.unique()==begin_index)[0][0]
        end_idx=np.argwhere(df.date.unique()==end_index)[0][0]
        if os.path.exists(self.npz_pwd):
            ary_dict = np.load(self.npz_pwd, allow_pickle=True)
            close_ary = ary_dict['close_ary']
            tech_ary = ary_dict['tech_ary']
        elif os.path.exists(self.csv_pwd): 
            tech_ary = list()
            close_ary = list()
            df_len = len(df.index.unique())  # df_len = max_step
            tic_list = df.tic.unique()
            print("Begin preparing the data")
            for day in tqdm(range(df_len)):
                item = df.loc[day]
                tech_items = [item.loc[item['tic']==tic,:][tech_id_list].values[0].tolist() for tic in tic_list]
                tech_items_flatten = sum(tech_items, [])
                tech_ary.append(tech_items_flatten)
                close_ary.append(item.close_price)

            close_ary = np.array(close_ary)
            tech_ary = np.array(tech_ary)

            np.savez_compressed(self.npz_pwd, close_ary=close_ary, tech_ary=tech_ary, )
            print("Data preparation Complete")
        else:
            error_str = "Not Found the DATA!"
            raise FileNotFoundError(error_str)
        
        Drelation = pd.read_pickle(self.relation_pwd)['cov'].values
        
        close_ary = close_ary[beg_idx:end_idx]
        tech_ary = tech_ary[beg_idx:end_idx]
        Drelation = Drelation[beg_idx:end_idx]
        return close_ary, tech_ary, Drelation
