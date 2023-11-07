import sys
import torch
import numpy as np
import pandas as pd
import os
from stock_env import StockTradingEnv
from stock_env import get_gym_env_args
from stock_env import build_env
from agent import Agent
from net import Actor
from train import train_agent
from train import evaluate_agent
import matplotlib.pyplot as plt

def load_torch_file(model, _path):
    state_dict = torch.load(_path)
    model.load_state_dict(state_dict)

class Arguments:
    def __init__(self, agent, env_func=None, env_args=None):
        self.env_func = env_func 
        self.env_args = env_args 

        self.max_step = self.env_args['max_step']
        self.env_name = self.env_args['env_name']
        self.state_dim = self.env_args['state_dim']
        self.action_dim = self.env_args['action_dim']

        self.agent = agent 
        self.state_input_dim = 64 
        self.action_dim = 30
        self.act_mid_layer_num=2
        self.act_mid_dim = 64
        self.cri_mid_layer_num=2
        self.cri_mid_dim = 64

        self.if_off_policy = False

        '''Arguments for device'''
        self.worker_num = 4 
        self.thread_num = 16
        self.random_seed = 42
        self.gpu_id = 1 

        '''Arguments for evaluate'''
        self.cwd = None  
        self.rescwd = None
        self.if_remove = True  
        self.epoch = 10
    def init_before_training(self):

        if self.cwd is None:
            self.cwd = f'DGDRL/{self.env_name}_{self.agent.__name__[5:]}_{self.gpu_id}'
            self.rescwd = 'DGDRL/result'

        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
            shutil.rmtree(self.rescwd, ignore_errors=True)
            print(f"| Arguments Remove result cwd: {self.rescwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)
        os.makedirs(self.rescwd, exist_ok=True)



def run():
    gpu_id = 1
    env = StockTradingEnv()
    env_func = StockTradingEnv
    env_args = get_gym_env_args(env=env, if_print=False)
    env_args['beg_idx'] = 0         # training begin and end
    env_args['end_idx'] = 3163    # DOWS:3272 SSE50:3163 NAS:3275
    args = Arguments(Agent, env_func=env_func, env_args=env_args)
    args.target_step = args.max_step
    args.epoch = 200
    args.gpu_id = gpu_id

    # train
    train_agent(args)


def evaluate_models_in_directory(dir_path=None, gpu_id=-1):
    print(f"| evaluate_models_in_directory: gpu_id {gpu_id}")
    print(f"| evaluate_models_in_directory: dir_path {dir_path}")

    model_names = [name for name in os.listdir(dir_path) if name[:6] == 'actor_']
    model_names.sort()
    env_func = StockTradingEnv
    env_args = {
            'env_name': 'StockTradingEnv-v2',
            'max_step': 3404,   # DOWS:3523 SSE50:3404    NAS:3524 
            'beg_idx': 3143,    # DOWS:3272(-20) SSE50:3163(-20) NAS:3275(-20)   
            'end_idx': 3404,    # DOWS:3523 SSE50:3404    NAS:3524     
        }
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    env = build_env(env_func=env_func, env_args=env_args)
    actor = Actor(mid_dim=64
                        , mid_layer_num=2
                        , state_dim=64
                        , action_dim=30
                        ).to(device)
    torch.set_grad_enabled(False)
    for model_name in model_names:
        
        
        model_path = f"{dir_path}/{model_name}"
        
        load_torch_file(actor, model_path)
        cumulative_returns,everyday_assets,rewards,daily_return = evaluate_agent(env, actor)
        everyday_assets = np.array(everyday_assets)
        rewards = np.array(rewards)
        daily_return = np.array(daily_return)
        final_res = pd.DataFrame({"everyday_assets":everyday_assets,"rewards":rewards,"daily_return":daily_return})
        print(final_res)
        final_res.to_csv(dir_path+"/"+str(model_names.index(model_name))+"trade_res.csv")
        print(f"cumulative_returns {cumulative_returns:9.3f}  {model_name}")
        #plot("NAS-Reward",rewards)
    

def plot(name, amount):
    plt.figure(figsize=(15, 6))
    plt.rcParams["font.size"] = 18

    plt.grid(visible=True, which="major", linestyle="-")
    plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5)
    plt.minorticks_on()

    plt.plot(range(len(amount)), amount, color="red", label="return", linewidth=3)

    plt.title("Backtest")
    plt.xlabel("Date")
    plt.ylabel("return")

    plt.legend()
    plt.savefig("DGDRL/result/" + name + "return.png")
    plt.close()


if __name__ == '__main__':
    # set seed
    random_seed=42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    thread_num = 16
    torch.set_num_threads(thread_num)
    torch.set_default_dtype(torch.float32)
    run()
    gpu_id = 1
    dir_path="DGDRL/StockTradingEnv-v2"
    evaluate_models_in_directory(dir_path, gpu_id)
