import torch
import torch.nn.functional as F
import torch.optim as optim
from env import *
from agent import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    #args
    n_step = 5

    env = make_atari_env('BeamRider-v0')
    obs = env.reset()

    policy = Policy(84, 84, 4, env.action_space.n)
    policy.eval()

    optimizer = optim.Adam(policy.parameters())

    for i in range(40):
        R = 0
        rollout = []
        optimizer.zero_grad()

        for i in range(n_step):
            action_prob, value = policy.forward(obs)
            action = choose_action(action_prob)
            print(action)
            print(action_prob)
            next_obs, reward, done, info = env.step(action)

            rollout.append((obs, reward, value, action_prob, action))
            obs = next_obs

            if done:
                break

        if done:
            R = 0
        else:
            _, _, R = policy.forward(obs)

        loss = 0
        for i in reversed(range(len(rollout))):
            obs_i, reward_i, value_i, action_prob_i, action_i = rollout[i]
            R += reward_i  # * gamma
            advantage_i = R - value_i.detach()
            policy_loss = - action_prob_i[[0], action[0]] * advantage_i # have to change this to use log
            value_loss = F.mse_loss(R, value_i)

            loss += (policy_loss + value_loss)
#            for param in policy.parameters():
#                param.grad.data.clamp_(-1, 1)
        loss.backward()

        # optimizer.step()



'''
    for i in range(1000):
        action_prob, value = policy.forward(obs)
        action = choose_action(action_prob)
        obs, reward, done, info = env.step(action)
        print(action)
        env.render()
        if done:
            print('------------------------------------------------------')
            env.reset()

'''
