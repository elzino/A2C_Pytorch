import torch
from tqdm import tqdm

from agent import Policy, choose_action

from vecenv.vev_env import make_env, VecToTensor
from vecenv.dummy_vec_env import DummyVecEnv

def main(load_path, num_episode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_env = 1

    env_id = 'Breakout-v0'
    envs = [make_env(env_id) for _ in range(n_env)]
    envs = DummyVecEnv(envs)
    envs = VecToTensor(envs)

    policy = Policy(84, 84, 4, envs.action_space.n).to(device)
    policy.load_state_dict(torch.load(load_path, map_location=device))
    policy.eval()

    for i in tqdm(range(num_episode)):
        obs = envs.reset()
        total_rewards = 0
        while True:
            action_logits, values = policy(obs)
            actions = choose_action(action_logits)

            next_obs, rewards, dones, info = envs.step(actions)
            total_rewards += rewards

            envs.render()

            if dones:
                break

        print('--------------------' + str(total_rewards.item())+'-------------------')

    envs.close()


if __name__ == '__main__':
    main('./tmp/06_02_02_05_at_97000.pt', 20)
