import cv2
import numpy as np
import d3dshot
import time
import pytesseract
import re
from PIL import Image

from keyboard import PressKey, ReleaseKey, W, A, S, D, R

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
DRIVE = True

if DRIVE:
  print("Focus game so that client can drive:")
  for i in list(range(4))[::-1]:
    print (i+1)
    time.sleep(1)

d = d3dshot.create("numpy")
d.display = d.displays[1]

def process_img(image):
  gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edge_img = cv2.Canny(gray_img, threshold1=150, threshold2=300)
  return edge_img

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

def read_time(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, img = cv2.threshold(img,254,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  cfg = r'-c tessedit_char_whitelist="0123456789:." --psm 6'
  return pytesseract.image_to_string(img, config=cfg)

def read_speed(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, img = cv2.threshold(img,254,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  cfg = r'-c tessedit_char_whitelist="0123456789" --psm 6'
  return pytesseract.image_to_string(img, config=cfg)

BOTTOM_CENTER_TIME = (655, 570, 45, 145)
BOTTOM_CENTER_SPEED = (590, 565, 75, 160)
TOP_CENTER_TIME = (190, 600, 30, 70)

start_time = time.time()
check_time = time.time()
down = False
prev_time = time.time()
while True:
  img = d.screenshot(region=(0, 690, 1270, 1410))
  lines_img = process_img(img)
  #print("Time Between Frame: ", time.time()-prev_time)
  prev_time = time.time()
  
  if time.time() - check_time > 0.5:
    print("lap_time:", read_time(select_region(img, TOP_CENTER_TIME)))
    print("time:", read_time(select_region(img, BOTTOM_CENTER_TIME)))
    print("speed:", read_speed(select_region(img, BOTTOM_CENTER_SPEED)))
    check_time = time.time()

  if DRIVE and not down:
    PressKey(W)
    down = True
    start_time = time.time()

  if DRIVE and time.time() - start_time > 10:
    ReleaseKey(W)
    down = False
    PressKey(R)
    time.sleep(1)
    ReleaseKey(R)