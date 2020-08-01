import cv2
import numpy as np
import d3dshot
import time
import pytesseract
from PIL import Image

# TODO: Replace this with a virtual controller
from keyboard import PressKey, ReleaseKey, W, A, S, D, R, Enter

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

BOTTOM_CENTER_TIME = (655, 570, 45, 145)
BOTTOM_CENTER_SPEED = (590, 565, 75, 160)
TOP_CENTER_TIME = (190, 600, 30, 70)

BOTTOM_LEFT_SCREEN_REGION = (0,690, 1270, 1410)


class TrackManiaInterface():
  def __init__(self, capture_box=BOTTOM_LEFT_SCREEN_REGION):
    self.capture = d3dshot.create("numpy")
    self.capture.display = self.capture.displays[1]
    self.capture_region = capture_box
    self.capture.capture(region=self.capture_region)
  
  def __del__(self):
    self.capture.stop()
  
  def disconnect(self):
    self.capture.stop()
    self.reset_keys()
  
  def show_capture(self):
    cv2.imshow(self.capture_screen())
  
  def capture_screen(self, count=1):
    return self.capture.get_frame_stack(list(range(count)), stack_dimension="last")
  
  def select_region(self, img, boundingBox):
    """
    selection region returns a sub_image given bounding_box tuple
    (yOffset, xOffset, height, width)
    """
    yOffset = boundingBox[0]
    xOffset = boundingBox[1]
    height = boundingBox[2]
    width = boundingBox[3]

    return img[yOffset:yOffset+height, xOffset:xOffset+width]
  
  def speed(self, img):
    return self.read_speed(
      self.select_region(img, BOTTOM_CENTER_SPEED))
  
  def time(self, img):
    return self.read_time(
      self.select_region(img, BOTTOM_CENTER_TIME))

  def lap_time(self, img):
    return self.read_time(
      self.select_region(img, TOP_CENTER_TIME))

  def read_time(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,254,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cfg = r'-c tessedit_char_whitelist="0123456789:." --psm 6'
    return pytesseract.image_to_string(img, config=cfg)

  def read_speed(self, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img,254,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cfg = r'-c tessedit_char_whitelist="0123456789" --psm 6'
    speed_string = pytesseract.image_to_string(img, config=cfg)
    value = 0
    try:
      value = int(speed_string)
    except ValueError:
      value = 0
    
    return value
  
  def forward(self):
    PressKey(W)

  def backward(self):
    PressKey(S)

  def left(self):
    PressKey(A)
  
  def right(self):
    PressKey(D)

  def reset_keys(self):
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    ReleaseKey(R)
  
  def reset(self):
    self.reset_keys()

    PressKey(R)
    PressKey(Enter)
    time.sleep(1/60)
    ReleaseKey(R)
    ReleaseKey(Enter)