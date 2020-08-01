import time

import cv2
import ctypes

from tesseract import Tesseract

BOTTOM_CENTER_TIME = (655, 570, 45, 145)
BOTTOM_CENTER_SPEED = (590, 565, 75, 160)
TOP_CENTER_TIME = (190, 600, 30, 70)

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

def read_speed(tess, img):
  proc_start = time.time()
  img = select_region(img, BOTTOM_CENTER_SPEED)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  print("image processing: {}".format(time.time() - proc_start))

  height, width = img.shape

  tess_start = time.time()
  tess.set_variable(b"tesseract_char_whitelist", b"0123456789")
  tess.set_variable(b"tessedit_pageseg_mode", b"6")
  tess.set_variable(b"user_defined_dpi", b"70")
  tess.set_image(img, width, height, 1)
  text = tess.get_utf8_text()
  print("tesseractapi: {}".format(time.time() - tess_start))
  return text.strip()

if __name__ == '__main__':
  img = cv2.imread('track1.png')
  start_time = time.time()
  tess = Tesseract(datapath=b"C:\Program Files\Tesseract-OCR\\tessdata")
  print("tesseract init: {}".format(time.time() - start_time))
  text = read_speed(tess, img)
  print("RESULT:", text)

