import cv2
import numpy as np

from tf_agents.trajectories import time_step as ts
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.specs import array_spec
from tf_agents.environments import utils
from timecode import seconds
import time

from tm_interface import TrackManiaInterface

ACTION_MAP = {
  0: "N-STRAIGHT",
  1: "N-LEFT",
  2: "N-RIGHT",
  3: "F-STRAIGHT",
  4: "F-LEFT",
  5: "F-RIGHT",
  6: "B-STRAIGHT",
  7: "B-LEFT",
  8: "B-RIGHT",
  9: "RESET",
}

TIMEOUT_OF_ATTEMPT = 30

class TrackManiaEnv(py_environment.PyEnvironment):
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=8, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(90, 128), dtype=np.uint8, minimum=0, maximum=255, name='observation')
    self._trackmania_interface = TrackManiaInterface()

  def action_spec(self):
    return self._action_spec
  
  def observation_spec(self):
    return self._observation_spec

  def _convert_to_obs(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dsize=(128,90), interpolation=cv2.INTER_AREA)
    return img
  
  def _reset(self):
    self.timer = time.time()
    self._trackmania_interface.reset()
    img = self._trackmania_interface.capture_screen()

    return ts.restart(self._convert_to_obs(img))

  def _step(self, action):
    self._apply_action(action)

    img = self._trackmania_interface.capture_screen()
    obs = self._convert_to_obs(img)

    speed = self._trackmania_interface.speed(img)
    time = self._trackmania_interface.time(img)

    time_sec = seconds(time)
    if time_sec > TIMEOUT_OF_ATTEMPT:
      return ts.termination(obs, speed)

    return ts.transition(
      obs, reward=speed, discount=0.3
    )

  def _apply_action(self, action):
    self._trackmania_interface.reset_keys()

    if action == 9:
      self._trackmania_interface.reset()
      return

    if 3 <= action < 6:
      self._trackmania_interface.forward()
    elif 6 <= action < 9:
      self._trackmania_interface.backward()
    
    if action % 3 == 1:
      self._trackmania_interface.left()
    elif action % 3 == 2:
      self._trackmania_interface.right()