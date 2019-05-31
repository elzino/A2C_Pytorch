import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from env import *
from agent import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Runner(object):
    def __init__(self, envs, policy, n_step, gamma):
        self.envs = envs
        self.policy = policy
        self.n_step = n_step
        self.n_env = envs.num_envs if hasattr(envs, 'num_envs') else 1
        obs_shp = envs.observation_space.shape
        self.mb_obs_shape = (self.n_env * n_step,) + obs_shp
        self.obs = envs.reset()
        self.dones = [False for _ in range(self.n_env)]
        self.gamma = gamma

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        for _ in range(self.n_step):
            with torch.no_grad():
                action_logits, values = self.policy.forward(self.obs)
                actions = choose_action(action_logits)
            next_obs, rewards, dones, info = self.envs.step(actions)

            mb_obs.append(self.obs)
            mb_dones.append(self.dones)
            mb_rewards.append(rewards)
            mb_actions.append(actions)
            mb_values.append(values)

            self.obs = next_obs
            self.dones = dones

        mb_dones.append(self.dones)

        mb_obs = torch.stack(mb_obs).transpose(1, 0)
        mb_actions = torch.stack(mb_actions).transpose(1, 0)
        mb_values = torch.stack(mb_values).transpose(1, 0)

        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]

        if self.gamma > 0:
            with torch.no_grad():
                last_values = self.policy.value(self.obs).cpu()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):

                if dones[-1] == 0:
                    rewards = discount_reward(np.append(rewards, value), np.append(dones, 0), self.gamma)[:-1]
                else:
                    rewards = discount_reward(rewards, dones, self.gamma)
                mb_rewards[n] = rewards

        mb_obs = mb_obs.contiguous().view(self.mb_obs_shape)
        mb_values = mb_values.flatten()
        mb_actions = mb_actions.flatten()
        mb_rewards = torch.from_numpy(mb_rewards).to(device).flatten()
        return mb_obs, mb_rewards, mb_values, mb_actions


def discount_reward(rewards, dones, gamma):
    discounted = []
    R = 0
    for r, done in zip(rewards[::-1], dones[::-1]):
        R = r + gamma * R * (1. - done)
        discounted.append(R)
    return discounted[::-1]
