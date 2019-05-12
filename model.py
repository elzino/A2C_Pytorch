# Reference : https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py

import torch


class Model(object):
    def __init__(self, policy, env, nsteps,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        pass

    def train(self, obs, rewards, actions, values, masks):
        pass


