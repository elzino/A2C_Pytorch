import gym
from gym.core import Wrapper
import time
from glob import glob
import csv
import os.path as osp
import json
import numpy as np

from vecenv.vev_env import VecEnvWrapper


class Monitor(Wrapper):
    EXT = "monitor.csv"

    def __init__(self, env, filename, allow_early_reset=False):
        super().__init__(env)
        self.tstart = time.time()
        if filename:
            self.result_writer = ResultWriter(filename)
        else:
            self.result_writer = None

        self.allow_early_reset = allow_early_reset
        self.rewards = None
        self.need_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0

    def reset(self, **kwargs):
        if not self.allow_early_reset and not self.need_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.need_reset = False
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.need_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.need_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            eptime = time.time() - self.tstart
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(eptime, 6)}

            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(eptime)

            if self.result_writer:
                self.result_writer.write_row(epinfo)

        self.total_steps += 1

    def close(self):
        if self.result_writer and self.result_writer.f:
            self.result_writer.f.close()


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None):
        super().__init__(venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        if filename:
            self.result_writer = ResultWriter(filename)
        else:
            self.result_writer = None

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1
        for i in range(len(dones)):
            if dones[i]:
                print(self.eprets[i])  # 나중에 지우기
                epinfo = {"r": round(self.eprets[i], 6), "l": self.eplens[i], "t": round(time.time() - self.tstart, 6)}
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
                if self.result_writer:
                    self.result_writer.write_row(epinfo)
        return obs, rews, dones, infos

    # close 구현?


class ResultWriter(object):
    def __init__(self, filename, header=''):
        assert filename is not None
        # 나중에 밑에 라인 지워 버리기, 별로 필요 없음
        if not filename.endswith(Monitor.EXT):
            if osp.isdir(filename):
                filename = osp.join(filename, Monitor.EXT)
            else:
                filename = filename + "." + Monitor.EXT
        self.f = open(filename, 'wt')
        if isinstance(header, dict):
            header = '# {} \n'.format(json.dump(header))

        self.f.write(header)
        self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't'))
        self.logger.writeheader()
        self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


