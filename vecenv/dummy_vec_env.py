import numpy as np
from .vev_env import VecEnv


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        obs_space = env.observation_space
        super().__init__(len(self.envs), obs_space, env.action_space)

        self.buf_obs = np.zeros((self.num_envs,) + obs_space.shape, dtype=obs_space.dtype)
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

        self.actions = None

    def step_async(self, actions):
        if len(actions) == self.num_envs:
            self.actions = actions
        else:
            assert self.num_envs == 1
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            self.buf_obs[e], self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(self.actions[e])
            if self.buf_dones[e]:
                self.buf_obs[e] = self.envs[e].reset()

        return np.copy(self.buf_obs), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()

    def reset(self):
        for e in range(self.num_envs):
            self.buf_obs[e] = self.envs[e].reset()
        return np.copy(self.buf_obs)

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)



