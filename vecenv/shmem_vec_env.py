import multiprocessing as mp
import numpy as np
import ctypes

from .vev_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars

_NP_TO_CT = {
    np.float32: ctypes.c_float,
    np.int32: ctypes.c_int32,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_char,
    np.bool: ctypes.c_bool
}


class ShmemVecEnv(VecEnv):
    def __init__(self, env_fns, context='spawn'):
        ctx = mp.get_context(context)

        dummy = env_fns[0]()
        observation_space, action_space = dummy.observation_space, dummy.action_space
        self.obs_dtype, self.obs_shape = observation_space.dtype.type, observation_space.shape
        dummy.close()

        super().__init__(len(env_fns), observation_space, action_space)
        self.obs_bufs = [ctx.Array(_NP_TO_CT[self.obs_dtype], int(np.prod(self.obs_shape))) for _ in env_fns]

        self.parent_pipes = []
        self.procs = []

        with clear_mpi_env_vars():
            for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker, args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shape, self.obs_dtype))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()

        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        for pipe in self.parent_pipes:
            pipe.recv()
        return self._read_obs()

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(('step', action))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        _, rews, dones, infos = zip(*outs)
        return self._read_obs(), np.array(rews), np.array(dones), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self):
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        return [pipe.recv() for pipe in self.parent_pipes]


    def _read_obs(self):
        obs = [np.frombuffer(buf.get_obj(), dtype=self.obs_dtype).reshape(self.obs_shape) for buf in self.obs_bufs]
        obs = np.array(obs)
        return obs


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_buf, obs_shape, obs_dtypes):
    def _write_obs(obs):
        dst = obs_buf.get_obj()
        dst_np = np.frombuffer(dst, dtype=obs_dtypes).reshape(obs_shape)
        np.copyto(dst_np, obs)

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                pipe.send((_write_obs(obs), reward, done, info))
            elif cmd == 'reset':
                pipe.send(_write_obs(env.reset()))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()



