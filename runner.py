import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from env import *
from agent import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Runner(object):
    def __init__(self, env, policy, n_step, gamma):
        self.env = env
        self.policy = policy
        self.n_step = n_step
        self.n_env = 1  #TODO modify this after implement of vecenv
        obs_shp = env.observation_space.shape  #  n_env obs_shp에 넣을지 말지
        self.mb_obs_shape = (self.n_env * n_step * obs_shp[0],) + obs_shp[1:]
        self.obs = env.reset()
        self.dones = [False for _ in range(self.n_env)]
        self.gamma = gamma

        self.reward_sum = 0

    #  vecenv로 바꿀때 []로 해놓은 것들 바꿔야함 reward, done
    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        for _ in range(self.n_step):
            action_probs, values = self.policy.forward(self.obs)
            actions = choose_action(action_probs)
            next_obs, reward, done, info = self.env.step(actions)
            self.reward_sum += reward

            mb_obs.append(self.obs)
            mb_dones.append(self.dones)
            mb_rewards.append([reward])
            mb_actions.append(actions)
            mb_values.append(values.detach())

            #  venv로 바꾸면 reset venv에다가 넣어야함
            if done:
                print(self.reward_sum)
                self.reward_sum = 0
                next_obs = self.env.reset()

            self.obs = next_obs
            self.dones = [done]

        mb_dones.append(self.dones)

        mb_obs = torch.stack(mb_obs).transpose(1, 0)# dtype 나중에 고민
        mb_actions = torch.stack(mb_actions).transpose(1, 0) # dtype 나중에 고민
        mb_rewards = torch.tensor(mb_rewards, dtype=torch.float32).transpose(1, 0)
        mb_values = torch.stack(mb_values).transpose(1, 0)# dtype 나중에 고민

        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0:
            last_values = self.policy.value(self.obs)
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()        # Without this code, the below array addition performs differently
                dones = dones.tolist()            # 이걸 안하면 밑에서 array 더하는거 연산이 다르게 됨

                if dones[-1] == 0:
                    rewards = discount_reward(rewards + [last_values], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_reward(rewards, dones, self.gamma)

                mb_rewards[n] = torch.tensor(rewards)

        mb_obs = mb_obs.view(self.mb_obs_shape).to(device)
        mb_rewards.flatten().to(device)
        mb_values.flatten().to(device)
        mb_actions.flatten().to(device)
        return mb_obs, mb_rewards, mb_values, mb_actions


def discount_reward(rewards, dones, gamma):
    discounted = []
    R = 0
    for r, done in zip(rewards[::-1], dones[::-1]):
        R = r + gamma * R * (1. - done)
        discounted.append(R)
    return discounted[::-1]
