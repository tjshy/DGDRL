from net import Actor, Critic
import torch
from typing import Tuple
from copy import deepcopy
import numpy as np
from tqdm import *
class Agent:
    def __init__(self
                , state_input_dim
                , act_mid_layer_num
                , act_mid_dim
                , cri_mid_layer_num
                , cri_mid_dim
                , action_dim
                , gpu_id=1
                ):
        self.if_off_policy = False
        self.act_class = Actor
        self.cri_class = Critic
        self.gamma = 0.99
        self.env_num = 1
        self.batch_size = 128
        self.repeat_times = 4
        self.reward_scale = 1
        self.learning_rate_cri = 0.08
        self.learning_rate_act = 0.08
        self.soft_update_tau = 2 ** -8

        self.if_off_policy = False
        self.if_act_target = False
        self.if_cri_target = False

        self.states = None  
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        
        self.state_input_dim = state_input_dim 
        self.action_dim = action_dim
        self.act_mid_layer_num=act_mid_layer_num
        self.act_mid_dim = act_mid_dim
        self.cri_mid_layer_num=cri_mid_layer_num
        self.cri_mid_dim = cri_mid_dim
        
        self.act = self.act_class(mid_dim=self.act_mid_dim
                                    , mid_layer_num=self.act_mid_layer_num
                                    , state_dim=self.state_input_dim
                                    , action_dim=self.action_dim
                                    ).to(self.device)
        self.cri = self.cri_class(mid_dim=self.cri_mid_dim
                                    , mid_layer_num=self.cri_mid_layer_num
                                    , state_dim=self.state_input_dim
                                    , action_dim=self.action_dim
                                    ).to(self.device) if self.cri_class else self.act

        self.act_target = deepcopy(self.act) if self.if_act_target else self.act
        self.cri_target = deepcopy(self.cri) if self.if_cri_target else self.cri

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate_cri)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate_act) \
            if self.cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()
        self.ratio_clip = 0.25
        self.lambda_entropy = 0.02  # 0.00~0.10

    def explore_env(self, env, target_step, soft_noise) -> list:
        traj_list = []
        last_done = [0, ]
        state = self.states[0][0]
        Drelation = self.states[0][1]

        step_i = 0
        done = False
        get_action = self.act.get_action
        while step_i < target_step:
            ten_s = torch.as_tensor(state, dtype=torch.float32)
            ten_dr = torch.as_tensor(Drelation, dtype=torch.float32)
            ten_a, ten_n = [ten.cpu() for ten in get_action(ten_s.to(self.device)
                                                            ,ten_dr.to(self.device)
                                                            ,soft_noise
                                                            )]
            next_s, reward, done, _ = env.step(ten_a[0].numpy())

            traj_list.append((ten_s, reward, done, ten_dr, ten_a, ten_n))

            step_i += 1
            if done:
                break
            else:
                state = next_s[0]
                Drelation = next_s[1]
        
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)
        
    def convert_trajectory(self, traj_list, last_done):
        traj_list = list(map(list, zip(*traj_list)))
        
        traj_list[0] = torch.stack(traj_list[0])
        traj_list[1] = (torch.tensor(traj_list[1], dtype=torch.float32) * self.reward_scale).unsqueeze(1)
        traj_list[2] = ((1 - torch.tensor(traj_list[2], dtype=torch.float32)) * self.gamma).unsqueeze(1)

        traj_list[3] = torch.stack(traj_list[3])

        traj_list[4:] = [torch.stack(item).squeeze(1) for item in traj_list[4:]]
        return traj_list
        
    def update_net(self, buffer):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_dr, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            buf_value=torch.zeros((buf_len,1), device=self.device)
            for i in range(buf_len):

                buf_value[i,0]=self.cri_target(buf_state[i], buf_dr[i]).squeeze(1)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)
            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            del buf_noise

        '''update network'''
        obj_critic = obj_actor = None
        update_times = int(1 + buf_len * self.repeat_times / self.batch_size)
        for train_step in tqdm(range(update_times)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            dr = buf_dr[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]

            action = buf_action[indices]
            logprob = buf_logprob[indices]


            new_logprob=torch.zeros(self.batch_size, device=self.device)
            obj_entropy=None
            for i in range(len(state)):
                new_logprob[i], _= self.act.get_logprob_entropy(state[i],action[i:i+1],dr[i])
            
            obj_entropy=(new_logprob.exp() * new_logprob).mean()
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, obj_actor)



            value = torch.zeros(self.batch_size, device = self.device)
            for i in range(len(state)):
                #TODO 加入图结构
                value[i] = self.cri(state[i],dr[i]).squeeze(1)
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)


        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), obj_actor.item(), a_std_log.item() 

    def get_reward_sum(self, buf_len, buf_reward, buf_mask, buf_value):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    @staticmethod
    def optimizer_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

