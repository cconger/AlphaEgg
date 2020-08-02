"""tm_env provides a training environment wrapping a running copy of trackmania"""
import numpy as np

from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.environments import utils

from environment import TrackManiaEnv

class TMAgentEnv(py_environment.PyEnvironment):
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=8, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(360, 128, 3), dtype=np.float32, minimum=0, maximum=1.0, name='observation')
    self._env = TrackManiaEnv()

  def __enter__(self):
    self._env.__enter__()
    return self

  def close(self):
    self._env.close()

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    obs = self._env.reset()
    return ts.restart(obs)

  def _step(self, action):
    if self._current_time_step.is_last():
      return self._reset()

    obs, reward, done, info = self._env.step(action)

    if done:
      return ts.termination(obs, reward)

    return ts.transition(obs, reward=reward, discount=0.3)