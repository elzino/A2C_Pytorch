import torch
import torch.nn.functional as F
import torch.optim as optim
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

    lr = 0.001
    alpha = 0.99
    epsilon = 1e-05

    env_id = 'Breakout-v0'
    envs = [make_env(env_id) for _ in range(n_env)]
#    envs = DummyVecEnv(envs)
#    envs = SubprocVecEnv(envs)
    envs = ShmemVecEnv(envs)
    envs = VecToTensor(envs)
    mon_file ="./tmp/" + datetime.now().strftime('%m_%d_%H_%M')
    envs = VecMonitor(envs, mon_file)

    train_policy = Policy(84, 84, 4, envs.action_space.n).to(device)
    step_policy = Policy(84, 84, 4, envs.action_space.n).to(device)
    step_policy.load_state_dict(train_policy.state_dict())
    step_policy.eval()

    runner = Runner(envs, step_policy, n_step, gamma)

    optimizer = optim.RMSprop(train_policy.parameters(), lr=lr, alpha=alpha, eps=epsilon)

    for i in tqdm(range(100)):
        mb_obs, mb_rewards, mb_values, mb_actions = runner.run()

        action_probs, values = train_policy(mb_obs)

        mb_adv = mb_rewards - mb_values
        neglogpac = - action_probs[np.arange(len(mb_actions)), mb_actions].log()
        pg_loss = torch.mean(neglogpac * mb_adv)

        vf_loss = F.mse_loss(values, mb_rewards)

        entropy = torch.mean(Categorical(action_probs).entropy())

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        optimizer.zero_grad()
        loss.backward()

        for name, param in train_policy.named_parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-max_grad_norm, max_grad_norm)

        optimizer.step()
        step_policy.load_state_dict(train_policy.state_dict())

