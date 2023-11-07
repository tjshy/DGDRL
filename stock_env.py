# stock.env

import os
import numpy as np
import pandas as pd
import numpy.random as rd
from tqdm import *
class StockTradingEnv:
    def __init__(self, initial_amount=1e6, gamma=0.99,
                 beg_idx=0, end_idx=3272):
        self.source = "SSE50"
        self.npz_pwd = 'DGDRL/data/'+self.source+'/'+self.source+'.numpy.npz'
        self.csv_pwd = 'DGDRL/data/'+self.source+'/'+self.source+'.csv'
        self.relation_pwd = 'DGDRL/data/'+self.source+'/'+'relation2.pkl'
        self.close_ary, self.tech_ary, self.Drelation = self.load_data_from_disk()
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]
        self.Drelation = self.Drelation[beg_idx:end_idx]
        print(f"| StockTradingEnv: close_ary.shape {self.close_ary.shape}")
        print(f"| StockTradingEnv: tech_ary.shape {self.tech_ary.shape}")
        print(f"| StockTradingEnv: Drelation.shape {self.Drelation.shape}")
        
        self.initial_amount = initial_amount
        self.gamma = gamma

        self.day = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = 0
        self.everyday_asset=[]
        self.daily_return = []

        self.window=20
        self.shares_num = self.close_ary.shape[1]# 股票数

        # environment information
        self.env_name = 'StockTradingEnv-v2'
        self.state_dim = (self.shares_num,self.window,80)
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

    def load_data_from_disk(self, tech_id_list=None):
        if self.source=="SSE50":
            tech_id_list = ["open_r","high_r","low_r","close_r","turnover_r","volume_r","macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
        else:
            tech_id_list = ["open_r","high_r","low_r","close_r","volume_r","macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

        if os.path.exists(self.npz_pwd):
            ary_dict = np.load(self.npz_pwd, allow_pickle=True)
            close_ary = ary_dict['close_ary']
            tech_ary = ary_dict['tech_ary']
        elif os.path.exists(self.csv_pwd):  # convert pandas.DataFrame to numpy.array
            df = pd.read_csv(self.csv_pwd,index_col = "Unnamed: 0")

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
                close_ary.append(item.close)

            close_ary = np.array(close_ary)
            tech_ary = np.array(tech_ary)

            np.savez_compressed(self.npz_pwd, close_ary=close_ary, tech_ary=tech_ary, )
            print("Data preparation Complete")
        else:
            error_str = "Not Found the DATA!"
            raise FileNotFoundError(error_str)
        
        Drelation = pd.read_pickle(self.relation_pwd)['cov'].values

        return close_ary, tech_ary, Drelation


def get_gym_env_args(env, if_print) -> dict:  # [ElegantRL.2021.12.12]

    import gym
    env_num = getattr(env, 'env_num') if hasattr(env, 'env_num') else 1

    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = getattr(env, 'env_name', None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        max_step = getattr(env, 'max_step', None)
        max_step_default = getattr(env, '_max_episode_steps', None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if not any(env.action_space.high - 1):
                print('WARNING: env.action_space.high', env.action_space.high)
            if not any(env.action_space.low - 1):
                print('WARNING: env.action_space.low', env.action_space.low)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info()'
                               '\n  Please set these value manually: if_discrete=bool, action_dim=int.'
                               '\n  And keep action_space in (-1, 1).')
    else:
        env_name = env.env_name
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args = {'env_num': env_num,
                'env_name': env_name,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(',', f",\n   ")
        env_args_repr = env_args_repr.replace('{', "{\n    ")
        env_args_repr = env_args_repr.replace('}', ",\n}")
        print(f"env_args = {env_args_repr}")
    return env_args

def kwargs_filter(func, kwargs: dict):
    import inspect
    sign = inspect.signature(func).parameters.values()
    sign = {val.name for val in sign}

    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_func=None, env_args=None):
    env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))
    return env