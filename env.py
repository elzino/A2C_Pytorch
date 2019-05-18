from collections import deque

import torch
import gym
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv, ShmemVecEnv, DummyVecEnv


# Code from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
               Reset only when lives are exhausted.
               This way all states are still reachable even though lives are episodic,
               and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=20):
        """Sample initial states by taking random number of no-ops on reset.
            No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2 : self._obs_buffer[0] = obs
            if i == self._skip - 1 : self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """
            Warp frames to 84x84 as done in the Nature paper and later work.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_colors, self._height, self._width),
            dtype=np.uint8,
        )

        original_space = self.observation_space
        self.observation_space = new_space

        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, observation):
        if self._grayscale:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(
            observation, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            observation = np.expand_dims(observation, 0)
        return observation


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

'''
This class is for replay buffer. But we don't use it so we don't need this class.

class LazyFrames(object):
    def __init__(self, frames):
        """
            This object ensures that common frames between the observations are only stored once.
            It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
            This object should only be converted to numpy array before being passed to the model.
            You'd not believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.as_type(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, item):
        return self._force()[..., i]  # i_th observation(frame) return
'''

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        """
            Stack k last frames
            I don't need lazy array here. So i changed it as numpy array.
        """
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[0] * k,) + shp[1:]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=0)


class ToTensor(gym.ObservationWrapper):
    '''
    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((1,) + shp), dtype=env.observation_space.dtype)
    '''
    def observation(self, observation):
        return torch.from_numpy(observation).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def make_atari_env(env_id):
    env = gym.make(env_id)
    env = NoopResetEnv(env)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # env = MaxAndSkipEnv(env) It is already applied by gym environment
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    return env


if __name__ == '__main__':
    # env test

    env = make_atari_env('Breakout-v0')
    # env = gym.make('BeamRider-v0')


    t = env.reset()
    v = env.reset()
    print(np.array_equal(t, v))

    mode = 'human'
    mode = 'rgb_array'

    s = env.render(mode)
    env.step(3)

    for i in range(100):
        o, r, d, _ = env.step(3)
        # print(d)
        env.render()
        if d:
            env.reset()
        #  s = input()
        #  print(i)

    print(o.shape)

# https://github.com/openai/baselines/blob/master/baselines/bench/monitor.py
