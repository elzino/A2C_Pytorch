import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical

from datetime import datetime
from tqdm import tqdm

from env import *
from agent import *
from runner import *
from monitor import VecMonitor
from vecenv.vev_env import make_env, VecToTensor
from vecenv.dummy_vec_env import DummyVecEnv
from vecenv.subproc_vec_env import SubprocVecEnv
from vecenv.shmem_vec_env import ShmemVecEnv

if __name__ == '__main__':
    #args
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_env = 2
    n_step = 5
    gamma = 0.99
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    save_model_per_epoch = 10
    num_updates = 100

    lr = 0.001
    alpha = 0.99
    epsilon = 1e-05

    env_id = 'Breakout-v0'
    envs = [make_env(env_id) for _ in range(n_env)]
#    envs = DummyVecEnv(envs)
#    envs = SubprocVecEnv(envs)
    envs = ShmemVecEnv(envs)
    envs = VecToTensor(envs)

    date = datetime.now().strftime('%m_%d_%H_%M')
    mon_file_name ="./tmp/" + date
    envs = VecMonitor(envs, mon_file_name)

    train_policy = Policy(84, 84, 4, envs.action_space.n).to(device)
    step_policy = Policy(84, 84, 4, envs.action_space.n).to(device)
    step_policy.load_state_dict(train_policy.state_dict())
    step_policy.eval()

    runner = Runner(envs, step_policy, n_step, gamma)

    optimizer = optim.RMSprop(train_policy.parameters(), lr=lr, alpha=alpha, eps=epsilon)

    for i in tqdm(range(num_updates)):
        mb_obs, mb_rewards, mb_values, mb_actions = runner.run()

        action_logits, values = train_policy(mb_obs)

        mb_adv = mb_rewards - mb_values
        dist = Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(mb_actions)
        pg_loss = torch.mean(-action_log_probs * mb_adv)

        vf_loss = F.mse_loss(values, mb_rewards)

        entropy = torch.mean(dist.entropy())

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(train_policy.parameters(), max_grad_norm)

#        for name, param in train_policy.named_parameters():
#            if param.grad is not None:
#                param.grad.data.clamp_(-max_grad_norm, max_grad_norm)

        optimizer.step()
        step_policy.load_state_dict(train_policy.state_dict())

        if i % save_model_per_epoch == 0 or i == (num_updates - 1):
            torch.save(train_policy.state_dict(), './models/{}_at_{}.pt'.format(date, i))

