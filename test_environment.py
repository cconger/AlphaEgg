import time
import numpy as np
import cv2

from environment import TrackManiaEnv

def format_millis(time):
  return "{:4.0f}ms".format(time * 1000)

with TrackManiaEnv(5) as env:

  per_episode_rewards = []
  per_episode_steps = []
  frame_times = []
  
  for i in range(10):
    print("Starting run", i)
    pre_reset_time = time.time()
    start = env.reset()

    cv2.imshow("input", start)
    cv2.waitKey(8)

    start_time = time.time()
    running = True
    steps = 0
    sum_reward = 0.
    while running:
      step_start = time.time()
      obs, reward, done, info = env.step(3)
      frame_times.append((time.time() - step_start)*1000)
      sum_reward += reward
      steps += 1
      if done:
        running = False

    finish = time.time()
    runtime = finish - start_time
    print("Finished")
    print("  Reward:", sum_reward)
    print("  Reset: ",  format_millis(start_time - pre_reset_time))
    print("  Time:  ", format_millis(runtime))
    print("  Steps: ", steps, "({:2.1f} per sec)".format(steps/runtime))
    per_episode_rewards.append(sum_reward)
    per_episode_steps.append(steps)
  
  print("Reward variance:")
  print("  Mean:", np.mean(per_episode_rewards))
  print("  Std :", np.std(per_episode_rewards))
  print("  Max :", np.max(per_episode_rewards))
  print("  Min :", np.min(per_episode_rewards))

  print("Frame variance:")
  print("  Mean:", np.mean(frame_times))
  print("  Std :", np.std(frame_times))
  print("  Max :", np.max(frame_times))
  print("  Min :", np.min(frame_times))
