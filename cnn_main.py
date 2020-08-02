import tensorflow as tf
import numpy as np
import time
import os

import random
import logging

from environment import TrackManiaEnv

n_actions = 9

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Memory:
  def __init__(self):
    self.clear()
  
  def clear(self):
    self.observations = []
    self.actions = []
    self.rewards = []
  
  def add_to_memory(self, obs, action, reward):
    self.observations.append(obs)
    self.actions.append(action)
    self.rewards.append(reward)

def choose_action(model, observation):
  observation = np.expand_dims(observation, axis=0)

  logits = model.predict(observation)

  prob_weights = tf.nn.softmax(logits).numpy()

  action = np.random.choice(n_actions, size=1, p=prob_weights.flatten())[0]
  
  return action

def create_network():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=300, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=n_actions, activation=None),
  ])

  return model

def normalize(x):
  x -= np.mean(x)
  x /= np.std(x)
  return x.astype(np.float32)

def discount_rewards(rewards, gamma=0.95):
  discounted_rewards = np.zeros_like(rewards)
  R = 0
  for t in reversed(range(0, len(rewards ))):
    R = R * gamma + rewards[t]
    discounted_rewards[t] = R
  
  return normalize(discounted_rewards)

def compute_loss(logits, actions, rewards):
  neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)

  loss = tf.reduce_mean(neg_logprob * rewards)

  return loss

def train_step(model, optimizer, observations, actions, discounted_rewards):
  with tf.GradientTape() as tape:
    logits = model(observations)

    loss = compute_loss(logits, actions, discounted_rewards)
  
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))


def run_training():
  # Should we load from our checkpoint directory
  LOAD_CHECKPOINT = False

  ### TRAINING
  learning_rate = 1e-4
  optimizer = tf.keras.optimizers.Adam(learning_rate)
  num_episodes = 1000
  checkpoint_model_interval = 100

  model = create_network()
  memory = Memory()
  rewards = []

  checkpoint_dir = "checkpoint"
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

  if LOAD_CHECKPOINT:
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).assert_consumed()

  with TrackManiaEnv() as env:
    #env.step = tf.function(env.step)

    for i_episode in range(num_episodes):
      print("Episodes {0} of {1}".format(i_episode + 1, num_episodes))
      observation = env.reset()
      memory.clear()

      while True:
        action = choose_action(model, observation)
        next_obs, reward, done, info = env.step(action)
        memory.add_to_memory(observation, action, reward)

        if done:
          total_reward = sum(memory.rewards)
          print("TOTAL REWARD:", total_reward)
          rewards.append(total_reward)

          train_step(
            model,
            optimizer,
            observations=np.stack(memory.observations, 0),
            actions=np.array(memory.actions),
            discounted_rewards= discount_rewards(memory.rewards)
          )

          memory.clear()
          break;
        observation = next_obs
      
      if i_episode > 0 and i_episode % checkpoint_model_interval == 0:
        print("Saving checkpoint!")
        checkpoint.save(file_prefix=checkpoint_prefix)

run_training()