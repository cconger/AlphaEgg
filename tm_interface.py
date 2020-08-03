"""tm_interface allows capture and key signaling to a running trackmania instance"""
import time

import cv2
import d3dshot
from tesseract import Tesseract

# TODO: Replace this with a virtual controller
from keyboard import PressKey, ReleaseKey, W, A, S, D, R, Enter

BOTTOM_LEFT_SCREEN_REGION = (0, 690, 1270, 1410)

class TrackManiaInterface():
  """TrackManiaInterface is a class for wrapping the capture and sending to a Trackmania process"""

  def __init__(self, capture_box=BOTTOM_LEFT_SCREEN_REGION):
    self.capture = d3dshot.create("numpy")
    self.capture.display = self.capture.displays[1]
    self.capture_region = capture_box
    self.capture.capture(region=self.capture_region)

  def __del__(self):
    self.capture.stop()

  def disconnect(self):
    self.capture.stop()
    ReleaseKeys(W, A, S, D, R)

  def capture_screen(self, count=1):
    return self.capture.get_frame_stack(list(range(count)), stack_dimension="first")
  
  def standard_actions(self, action):
    if action == 0:
      # No input
      ReleaseKeys(W, A, S, D, R)
    elif action == 1:
      # Neutral Left
      ReleaseKeys(W, S, D, R)
      PressKeys(A)
    elif action == 2:
      # Neutral Right
      ReleaseKeys(W, A, S, R)
      PressKeys(D)
    elif action == 3:
      # Forward
      ReleaseKeys(A, S, D, R)
      PressKeys(W)
    elif action == 4:
      # Forward Left
      ReleaseKeys(S, D, R)
      PressKeys(W, A)
    elif action == 5:
      # Forward Right
      ReleaseKeys(A, S, R)
      PressKeys(W, D)
    elif action == 6:
      # Backwards
      ReleaseKeys(W, A, D, R)
      PressKeys(S)
    elif action == 7:
      # Bwards Left
      ReleaseKeys(W, D, R)
      PressKeys(S, A)
    elif action == 8:
      # Backwards Right
      ReleaseKeys(W, A, R)
      PressKeys(S, D)

  def reset(self):
    ReleaseKeys(W, A, S, D, R)

    PressKey(R)
    PressKey(Enter)
    time.sleep(1.5)
    ReleaseKey(R)
    ReleaseKey(Enter)

def ReleaseKeys(*args):
  for arg in args:
    ReleaseKey(arg)

def PressKeys(*args):
  for arg in args:
    PressKey(arg)