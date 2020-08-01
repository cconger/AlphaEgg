from tm_env import TrackManiaEnv

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

import matplotlib
import matplotlib.pyplot as plt

import time
import os

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.enable_v2_behavior()

## Hyperparams
num_iterations = 10000

collect_steps_per_iteration = 5
replay_buffer_max_length = 10000

batch_size = 64
learning_rate = 1e-3
log_interval = 50

num_eval_episodes = 5
eval_interval = 5000

def run():
  print("Focus game so that client can drive:")
  for i in list(range(4))[::-1]:
    print (i+1)
    time.sleep(1)
    
  env = TrackManiaEnv()
  tf_env = tf_py_environment.TFPyEnvironment(env)
  fc_layer_params = (100,)
  q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=fc_layer_params
  )

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

  global_counter = tf.compat.v1.train.get_or_create_global_step()
  #train_step_counter = tf.Variable(0)

  agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_counter,
  )

  root_dir = os.path.abspath(os.getcwd())
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')

  agent.initalize()

  train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
  ]

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_max_length,
  )

  collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_steps=collect_steps_per_iteration,
  )

  train_checkpointer = common.Checkpointer(
    ckpt_dir=train_dir,
    agent=agent,
    global_step=global_counter,
    metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
  )

  policy_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(train_dir, 'policy'),
    policy=agent.policy,
    global_step=global_counter,
  )

  rb_checkpointer = common.Checkpointer(
    ckpt_dir=os.path.join(train_dir,'replay_buffer'),
    max_to_keep=1,
    replay_buffer=replay_buffer,
  )

  train_checkpointer.initialize_or_restore()
  rb_checkpointer.initialize_or_restore()

  collect_driver.run = common.function(collect_driver.run)
  agent.train = common.function(agent.train)

  random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

  dynamic_step_driver.DynamicStepDriver(
    tf_env,
    random_policy,
    observers=[replay_buffer.add_batch] + train_metrics,
    num_steps=initial_collect_steps
  ).run()

  train_summary_writer = tf.summary.create_file_writer(train_dir)
  train_summary_writer.set_as_default()

  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]
  results = metric_utils.eager_compute(
    eval_metrics,
    tf_env,
    agent.policy,
    num_episodes=num_eval_episdoes,
    train_step=global_counter,
    summary_writer=tf.create_file_writer(eval_dir),
    summary_prefix='Metrics',
  )

  metric_utils.log_metrics(eval_metrics)

  time_step = None
  policy_state = collect_policy.get_initial_state(tf_env.batch_size)

  timed_at_step = global_counter.numpy()
  time_acc = 0

  dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=train_sequence_length + 1
  ).prefetch(3)

  iterator = iter(dataset)

  @common.function
  def train_step():
    experience, _ = next(iterator)
    return agent.train(experience)

  for _ in range(num_iterations):
    start_time = time.time()
    time_step, policy_state = collect_driver.run(
      time_step=time_step,
      policy_state=policy_state,
    )

    for _ in range(train_steps_per_iteration):
      train_loss = train_step()
    time_acc += time.time() - start_time

    step = global_counter.numpy()

    if step % log_interval == 0:
      logging.info("step = %d, loss = %f", step, train_loss.loss)
      steps_per_sec = (step - timed_at_step) / time_acc
      logging.info("%.3f steps/sec", steps_per_sec)
      timed_at_step = step
      time_acc = 0
    
    for train_metric in train_metrics:
      train_metric.tf_summaries(train_step=global_step, step_metrics=train_metrics[:2])
    
    if step % train_checkpoint_interval == 0:
      train_checkpointer.save(global_step=step)
    
    if step % policy_checkpoint_interval == 0:
      policy_checkpointer.save(global_step=step)
    
    if step % rb_checkpoint_interval == 0:
      rb_checkpointer.save(global_step=step)
    
    if step % eval_interval == 0:
      results = metric_utils.eager_compute(
        eval_metrics,
        tf_env,
        agent.policy,
        num_episodes=num_eval_episodes,
        train_step=global_step,
        summary_writer=eval_summary_writer,
        summary_prefix='Metrics',
      )
    return train_loss

  ### OLD BELOW
  print("COMPUTING CURRENT AVG RETURN:")
  avg_return = compute_avg_return(tf_env, agent.policy, num_eval_episodes)
  print("Return: ", avg_return)
  returns = [avg_return]

  for _ in range(num_iterations):

    for _ in range(collect_steps_per_iteration):
      collect_step(tf_env, agent.collect_policy, replay_buffer)
    
    experience, unused_info = next(ds_iter)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, train_loss))
    
    if step % eval_interval == 0:
      print('Running new eval, computing avg return')
      avg_return = compute_avg_return(tf_env, agent.policy, num_eval_episodes)
      print('step = {0}: Average Return = {1}'.format(step, avg_return))
      returns.append(avg_return)
  
  iterations = range(0, num_iterations + 1, eval_interval)
  plt.plot(iterations, returns)
  plt.ylabel('Average Returns')
  plt.xlabel('Iterations')
  plt.show(block=True)

run()