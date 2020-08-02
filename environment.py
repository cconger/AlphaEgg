"""provides a non tf-agents implementation of an environment around trackmania"""
import time

import cv2
import re
import numpy as np

from timecode import seconds

from tm_interface import TrackManiaInterface
from tesseract import Tesseract

# Screen Regions for important info
BOTTOM_CENTER_TIME = (655, 570, 45, 145)
BOTTOM_CENTER_SPEED = (590, 565, 75, 160)
TOP_CENTER_TIME = (190, 600, 30, 70)

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
}

TIMEOUT_OF_ATTEMPT = 15

AREA_OF_INTEREST = (160, 0, 400, 1280)

def format_frame(img):
  img = cv2.resize(select_region(img, AREA_OF_INTEREST), dsize=(128,90), interpolation=cv2.INTER_LINEAR)
  return img

class TrackManiaEnv(object):
  def __init__(self, run_timeout=TIMEOUT_OF_ATTEMPT):
    self._run_timeout = run_timeout
    self._trackmania_interface = TrackManiaInterface()
    self._tess = Tesseract(datapath=b"C:\Program Files\Tesseract-OCR\\tessdata")
  
  def __enter__(self):
    print("Focus game client")
    for i in reversed(range(3)):
      print("{0}...".format(i+1))
      time.sleep(1)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def close(self):
    self._trackmania_interface.disconnect()

  def _convert_to_obs(self, frames):
    return (np.concatenate([format_frame(fr) for fr in frames], axis=0)/255.).astype(np.float32)
  
  def action_space(self):
    return len(ACTION_MAP)

  def read_text_from_image(self, img, whitelist):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    height, width = img.shape
    self._tess.set_variable(b"tessedit_char_whitelist", whitelist)
    self._tess.set_variable(b"tessedit_pageseg_mode", b"6")
    self._tess.set_variable(b"user_defined_dpi", b"70")
    self._tess.set_image(img, width, height, 1)
    return self._tess.get_utf8_text().decode('utf-8')

  def read_speed(self, img):
    img = select_region(img, BOTTOM_CENTER_SPEED)
    text = self.read_text_from_image(img, b"0123456789")
    m = re.match("\d\d\d", text)
    if m is None:
      try:
        return int(text), False, text
      except ValueError:
        return 0, False, text
    return int(m.group(0)), True, text
  
  def read_time(self, img):
    img = select_region(img, BOTTOM_CENTER_TIME)
    text = self.read_text_from_image(img, b"0123456789:.")
    return text

  def reset(self):
    """
    Reset restarts the course and returns the starting observation
    """
    self.timer = time.time()
    self._trackmania_interface.reset()
    self._prev_speed = 0
    frames = self._trackmania_interface.capture_screen(4)

    return self._convert_to_obs(frames)

  def step(self, action):
    self._trackmania_interface.standard_actions(action)
    time.sleep(1/10)
    frames = self._trackmania_interface.capture_screen(4)
    img = frames[0]
    
    obs = self._convert_to_obs(frames)

    clock_time = self.read_time(frames[0])
    time_sec = seconds(clock_time)

    # Parse speeds from all frames
    speeds = [self.read_speed(frame) for frame in frames]
    if time_sec > 2 and np.sum([s[0] for s in speeds]) <= 200:
      # If we've been stuck at 0 for all frames after the first second... terminate
      return obs, -1., True, clock_time
    
    speed = 0
    for s in speeds: 
      if s[1]:
        speed = s[0]
        break

    reward = float(speed) / 300.
    if time_sec > self._run_timeout:
      return obs, reward, True, clock_time

    return obs, reward, False, clock_time


def select_region(img, boundingBox):
  """
  selection region returns a sub_image given bounding_box tuple
  (yOffset, xOffset, height, width)
  """
  yOffset = boundingBox[0]
  xOffset = boundingBox[1]
  height = boundingBox[2]
  width = boundingBox[3]

  return img[yOffset:yOffset+height, xOffset:xOffset+width]
