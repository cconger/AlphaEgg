import time

import cv2
from environment import TrackManiaEnv

SHOW_INPUTS = False

with TrackManiaEnv() as env:
  print("Focus game client")
  for i in reversed(range(3)):
    print("{0}...".format(i+1))
    time.sleep(1)

  start_obs = env.reset()
  if SHOW_INPUTS:
    cv2.imshow("RESET", start_obs)
    cv2.waitKey(1000)

  done = False
  start = time.time()
  frames = 0
  while not done:
    frames += 1
    next_obs, reward, done, info = env.step(3)
    if SHOW_INPUTS:
      cv2.imshow("UPDATE", next_obs)
      cv2.waitKey(8)
    print(time.time(), info, reward)
  
  print("Steps/sec:", frames/(time.time() - start))
  print(time.time(), "FINISHED", info)