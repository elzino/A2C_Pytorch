import multiprocessing as mp

import torch
import numpy as np
from .vev_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    print('worker running')
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send((env.observation_space, env.action_space, env.spec))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got keyboardInturreupt')
    finally:
        env.close()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, context='spawn'):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                                                    for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True
            with clear_mpi_env_vars():
                p.start()

        for work_remote in self.work_remotes:
            work_remote.close()

        self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv()
        self.viewer = None
        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        print(obs)
        return torch.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        assert not self.closed
        for remote in self.remotes:
            remote.send(('reset', None))
        return torch.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.close = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
        for remote in self.remotes:
            remote.send(('render', None))
        imgs = [remote.recv() for remote in self.remotes]
        return imgs

    def __del__(self):
        if not self.closed:
            self.close()






