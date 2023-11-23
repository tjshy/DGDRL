import sys
import torch
import numpy as np
import pandas as pd
import os
from stock_env import StockTradingEnv
from agent import AgentPPO
from net import Actor
from train import train_agent
from train import evaluate_agent
import matplotlib.pyplot as plt
from config import Config,result_path


def load_torch_file(model, _path):
    state_dict = torch.load(_path)
    model.load_state_dict(state_dict)

class Arguments:
    def __init__(self, agent, cfg):

        self.env_name = cfg.env_name
        self.agent = agent 
        self.state_input_dim = cfg.state_input_dim
        self.action_dim = cfg.action_dim
        self.act_mid_layer_num=cfg.act_mid_layer_num
        self.act_mid_dim = cfg.act_mid_dim
        self.cri_mid_layer_num=cfg.cri_mid_layer_num
        self.cri_mid_dim = cfg.cri_mid_dim
        self.if_off_policy = self.get_if_off_policy()
        self.thread_num = cfg.thread_num
        self.epoch = cfg.epoch 
        self.cwd = None  
        self.rescwd = None
        self.if_remove = True  
        self.gpu_id=cfg.gpu_id

    def init_before_training(self):

        if self.cwd is None:
            self.cwd = result_path+f'{self.env_name}_{self.agent.__name__[5:]}_{self.gpu_id}'
            self.rescwd = result_path+'result'

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

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find('PPO') == -1, name.find('A2C') == -1)) 

def run(cfg):
    env = StockTradingEnv(cfg,mode='train')
    args = Arguments(AgentPPO,cfg)
    args.target_step = env.max_step
    # 训练
    train_agent(args,env)

def evaluate_models_in_directory(dir_path=None, gpu_id=-1, cfg=None):
    print(f"| evaluate_models_in_directory: gpu_id {gpu_id}")
    print(f"| evaluate_models_in_directory: dir_path {dir_path}")

    model_names = [name for name in os.listdir(dir_path) if name[:6] == 'actor_']
    model_names.sort()
    env = StockTradingEnv(cfg,mode='test')
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    actor = Actor(mid_dim=64
                        , mid_layer_num=2
                        , state_dim=64
                        , action_dim=30
                        ).to(device)
    torch.set_grad_enabled(False)
    os.makedirs(dir_path,exist_ok=True)
    os.makedirs(result_path+'act', exist_ok=True)
    os.makedirs(result_path+"result", exist_ok=True)
    for model_name in model_names:
        
        model_path = f"{dir_path}/{model_name}"
        load_torch_file(actor, model_path)
        cumulative_returns,everyday_assets,rewards,daily_return,daily_action,D_weights, S_weights= evaluate_agent(env, actor)

        everyday_assets = np.array(everyday_assets)
        rewards = np.array(rewards)
        daily_return = np.array(daily_return)
        
        np.save(result_path+'act/'+str(model_names.index(model_name))+"action.npy",daily_action)
        np.save(result_path+'act/'+str(model_names.index(model_name))+"_D_weights.npy",D_weights)
        np.save(result_path+'act/'+str(model_names.index(model_name))+"_S_weights.npy",S_weights)
        
        final_res = pd.DataFrame({"everyday_assets":everyday_assets,"rewards":rewards,"daily_return":daily_return})
        print(final_res)
        final_res.to_csv(result_path+'act/'+str(model_names.index(model_name))+"trade_res.csv")
        print(f"cumulative_returns {cumulative_returns:9.3f}  {model_name}")
        plot("SSE-Reward",rewards)
    
def plot(name, amount):
    plt.figure(figsize=(15, 6))
    plt.rcParams["font.size"] = 18

    plt.grid(visible=True, which="major", linestyle="-")
    plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5)
    plt.minorticks_on()

    plt.plot(range(len(amount)), amount, color="red", label="return", linewidth=3)

    plt.title("PPO Backtest")
    plt.xlabel("Date")
    plt.ylabel("return")

    plt.legend()
    plt.savefig(result_path+"result/" + name + "return.png")
    plt.close()


if __name__ == '__main__':
    # set seed
    cfg=Config()
    random_seed=cfg.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    thread_num = cfg.thread_num
    torch.set_num_threads(thread_num)
    torch.set_default_dtype(torch.float32)
    run(cfg)
    gpu_id = cfg.gpu_id
    dir_path=result_path+"StockTradingEnv-v2_PPO_1"
    evaluate_models_in_directory(dir_path, gpu_id, cfg=cfg)
