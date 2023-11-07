import torch
import time
import os
from stock_env import build_env
from stock_env import StockTradingEnv
from replay_buffer import ReplayBufferList
import matplotlib.pyplot as plt
import numpy as np
def plot(name, amount):
    # 画图
    plt.figure(figsize=(15, 6))
    plt.rcParams["font.size"] = 18

    plt.grid(visible=True, which="major", linestyle="-")
    plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5)
    plt.minorticks_on()

    plt.plot(range(len(amount)), amount, color="red", label="return", linewidth=2)

    plt.title("train rewards")
    plt.xlabel("Date")
    plt.ylabel("return")

    plt.legend()
    plt.savefig("DGDRL/result/" + name)
    plt.close()

def train_agent(args):
    torch.set_grad_enabled(False)
    args.init_before_training()
    '''env-init'''
    env = build_env(args.env_func, args.env_args)

    agent = args.agent(args.state_input_dim
                , args.act_mid_layer_num
                , args.act_mid_dim
                , args.cri_mid_layer_num
                , args.cri_mid_dim
                , args.action_dim
                , args.gpu_id)
    
    buffer = ReplayBufferList()

    '''start training'''
    cwd = args.cwd
    target_step = args.target_step
    total_step = 0
    actor_loss=[]
    critic_loss=[]
    agent.states = [env.reset(), ]
    for i in range(args.epoch):
        trajectory = agent.explore_env(env, target_step, (1-i/args.epoch))#(1-i/args.epoch)
        #plot(f"epoch{i}_train_reward.png",env.everyday_asset)
        agent.states[0]=env.reset()
        steps, r_exp = buffer.update_buffer((trajectory,))

        # 开始训练
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        total_step += steps
        print(
                f"epoch:{i}  "
                f"Step:{total_step:8.2e}  "
                f"AvgReturn:{r_exp:8.2f}  "
                f"Returns:{env.cumulative_returns:8.2f}  "
                f"ObjC:{logging_tuple[0]:8.2f}  "
                f"ObjA:{logging_tuple[1]:8.2f}  "
            )
        actor_loss.append(logging_tuple[1])
        critic_loss.append(logging_tuple[0])
        e_return = online_evaluate_agent(i, agent.act)
        save_path = f"{cwd}/actor_epoch{i}_cumulationReturn{e_return:06.3f}.pth"
        torch.save(agent.act.state_dict(), save_path)
        

   

    
def online_evaluate_agent(epoch, actor):
    gpu_id = 1
    env_func = StockTradingEnv
    env_args = {
            'env_name': 'StockTradingEnv-v2',
            'max_step': 3524,   # DOWS:3523 SSE50:3404    
            'beg_idx': 3255,    # DOWS:3272(-20) SSE50:3163(-20) NAS:3275(-20)   
            'end_idx': 3524,    # DOWS:3523 SSE50:3404    NAS:3524
        }
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    env = build_env(env_func=env_func, env_args=env_args)
    amount = []
    amount.append(env.initial_amount)
    max_step = env.max_step
    state = env.reset()
    Drelation = state[1]
    state = state[0]
    episode_return = 0.0  # sum of rewards in an episode
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        dr_tensor = torch.as_tensor(Drelation, dtype=torch.float32, device=device)
        a_tensor = actor(s_tensor, dr_tensor)
        action = a_tensor.detach().cpu().numpy()[0] 
        state, reward, done, _ = env.step(action)     
        Drelation = state[1]
        state = state[0]
        episode_return += reward
        amount.append(env.total_asset)
        if done:
            break
    
    episode_step += 1
    e_return = env.cumulative_returns
    print("evaluate: ",e_return)
    plot(f"epoch{epoch}_evaluate_reward.png", amount)
    del env
    return e_return

def evaluate_agent(env, actor):
    max_step = env.max_step
    device = next(actor.parameters()).device
    state = env.reset()
    Drelation = state[1]
    state = state[0]
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        dr_tensor = torch.as_tensor(Drelation, dtype=torch.float32, device=device)
        a_tensor = actor(s_tensor, dr_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  
        state, reward, done, _ = env.step(action)

        Drelation = state[1]
        state = state[0]
        if done:
            break
    episode_step += 1
    # 每日资产总额列表 每日reward列表 每日日收益率
    return env.cumulative_returns, env.everyday_asset, env.rewards, env.daily_return


