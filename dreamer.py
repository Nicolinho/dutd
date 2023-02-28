import argparse
import collections
import functools
import math
import os
import pathlib
import sys
import time
import warnings


warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*TensorFloat-32 matmul/conv*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import wrappers
from plotting import load_jsonl


class Dreamer(tools.Module):

  def __init__(self, config, logger, dataset, cross_val_eps):
    self._config = config
    self._logger = logger
    self._float = prec.global_policy().compute_dtype
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(config.traindir), dtype=tf.int64)
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._dataset = iter(dataset)
    self._wm = models.WorldModel(self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._wm, config.behavior_stop_grad)
    reward = lambda f, s, a: self._wm.heads['reward'](f).mode()
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()
    # Train step to initialize variables including optimizer statistics.
    self._train(next(self._dataset))

    self.cross_val_eps = cross_val_eps
    self._metrics_cross_val = collections.defaultdict(tf.metrics.Mean)
    self._utd_ratio = config.train_every
    self._cross_val_last_loss = None
    self._should_update_utd = tools.Every(config.update_utd_ratio_every)
    self.should_sample_new_cv_data = tools.Every(config.crossval_new_dataset_every)
    self.should_sample_new_cv_data._last = 0
    self._should_log_cv = tools.Every(max(config.update_utd_ratio_every,
                                         config.log_every))
    # skip first log as the value have a much higher scale than the rest
    self._should_log_cv._last = config.prefill + self._should_log_cv._every
    self._crossval_only_record_loss = config.crossval_only_record_loss

    # init utd update variables
    if self._config.do_utd_updates:
      cross_val_dataset = iter(make_dataset(self.cross_val_eps, self._config,
                                            crossval=True))
      self._adjust_utd(cross_val_dataset, init=True)

  def __call__(self, obs, reset, state=None, training=True):
    step = self._step.numpy().item()
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      state = tf.nest.map_structure(lambda x: x * mask, state)
    if training and self._config.do_utd_updates and self._should_update_utd(step):
      cross_val_dataset = iter(make_dataset(self.cross_val_eps, self._config,
                                            crossval=True))
      self._adjust_utd(cross_val_dataset)
    elif training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)

      for _ in range(steps):
        self._train(next(self._dataset))
      if self._should_log_cv(step):
        for name, mean in self._metrics_cross_val.items():
          self._logger.scalar(name, float(mean.result()))
          mean.reset_states()
      if self._should_log(step):
        for name, mean in self._metrics.items():
          self._logger.scalar(name, float(mean.result()))
          mean.reset_states()
        openl = self._wm.video_pred(next(self._dataset))
        self._logger.video('train_openl', openl)
        self._logger.write(fps=True)
    policy_output, state = self._policy(obs, state, training)
    if training:
      self._step.assign_add(len(reset))
      self._logger.step = self._config.action_repeat \
          * self._step.numpy().item()
    return policy_output, state

  @tf.function
  def _policy(self, obs, state, training):
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      action = tf.zeros((batch_size, self._config.num_actions), self._float)
    else:
      latent, action = state
    embed = self._wm.encoder(self._wm.preprocess(obs))
    latent, _ = self._wm.dynamics.obs_step(
        latent, action, embed, self._config.collect_dyn_sample)
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat = self._wm.dynamics.get_feat(latent)
    if not training:
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
    elif self._should_expl(self._step):
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
    logprob = actor.log_prob(tf.cast(action, tf.float32))
    if self._config.actor_dist == 'onehot_gumble':
      action = tf.cast(
          tf.one_hot(tf.argmax(action, axis=-1), self._config.num_actions),
          action.dtype)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    amount = tf.cast(amount, self._float)
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      return tools.OneHotDist(probs=probs).sample()
    else:
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  @tf.function
  def _train(self, data):
    print('Tracing train function.')
    metrics = {}
    post, context, mets = self._wm.train(data)
    metrics.update(mets)
    start = post
    if self._config.pred_discount:  # Last step could be terminal.
      start = {k: v[:, :-1] for k, v in post.items()}
      context = {k: v[:, :-1] for k, v in context.items()}
    reward = lambda f, s, a: self._wm.heads['reward'](
        self._wm.dynamics.get_feat(s)).mode()
    metrics.update(self._task_behavior.train(start, reward)[-1])
    if self._config.expl_behavior != 'greedy':
      if self._config.pred_discount:
        data = {k: v[:, :-1] for k, v in data.items()}
      mets = self._expl_behavior.train(start, context, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    for name, value in metrics.items():
      self._metrics[name].update_state(value)

  @tf.function
  def _get_loss(self, data_batch):
    print('Tracing adjust_utd function.')
    return self._wm.train(data_batch, only_losses=True)

  def _adjust_utd(self, cross_val_dataset, init=False):
    metrics = {}
    for i, data in enumerate(cross_val_dataset):
      post, context, mets = self._get_loss(data)

      if not metrics:
        metrics.update(mets)
      else:
        for key, value in mets.items():
          metrics[key] += value

    print("NUM UTD BATCHES: ", i)

    for name, value in metrics.items():
      self._metrics_cross_val['crossval' + name].update_state(value / (i+1))

    if (self._cross_val_last_loss is not None) and (not init) and (not self._crossval_only_record_loss):
      old_ceiled_utd_ratio = math.ceil(self._utd_ratio)
      if self._cross_val_last_loss > metrics['image_loss'].numpy():
        # the old val loss was higher than the new one, so model is
        # not overfitting and more train steps per env step can be performed

        self._utd_ratio = min(self._config.max_utd_ratio,
                              max(self._utd_ratio / self._config.update_utd_increment,
                                  self._config.min_utd_ratio))
      else:
        self._utd_ratio = min(self._config.max_utd_ratio,
                              max(self._utd_ratio * self._config.update_utd_increment,
                                  self._config.min_utd_ratio))

      if math.ceil(self._utd_ratio) != old_ceiled_utd_ratio:
        self._should_train._every = math.ceil(self._utd_ratio)

    self._metrics_cross_val['crossval_utd_ratio'].update_state(self._utd_ratio)

    self._cross_val_last_loss = metrics['image_loss'].numpy()




def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config, crossval = False):
  example = episodes[next(iter(episodes.keys()))]
  types = {k: v.dtype for k, v in example.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in example.items()}
  # as dreamer always trains on a chunk from one episode the dataset generator
  #  randomly chooses one episode and takes batch_length number of states
  #  from that episode
  if crossval:
    generator = lambda: tools.sample_episodes_crossval(
        episodes, config.batch_length, config.oversample_ends)
  else:
    generator = lambda: tools.sample_episodes(
        episodes, config.batch_length, config.oversample_ends)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.prefetch(10)
  return dataset


def make_env(config, logger, mode, train_eps, eval_eps, crossval_eps=None):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, config.size,
        grayscale=config.grayscale,
        life_done=False and (mode in ['train', 'crossval']),
        sticky_actions=config.sticky_actions,
        all_actions=True)
    env = wrappers.OneHotAction(env)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit)
  env = wrappers.SelectAction(env, key='action')
  callbacks = [functools.partial(
      process_episode, config, logger, mode, train_eps, eval_eps, crossval_eps)]
  env = wrappers.CollectDataset(env, callbacks)
  env = wrappers.RewardObs(env)
  return env


def process_episode(config, logger, mode, train_eps, eval_eps, crossval_eps,
                    episode):
  # this gets called after the end of an episode from the wrappend env
  # and the complete episode is
  # then stored in either traindir or evaldir (depending on which env called it)
  #
  directory = dict(train=config.traindir, eval=config.evaldir,
                   crossval=config.crossval_dir)[mode]
  cache = dict(train=train_eps, eval=eval_eps, crossval=crossval_eps)[mode]
  filename = tools.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()

  # here old episodes are deleted from the train_eps dict if the buffer size
  # is reached. However, I think they are not deleted from the folder, where
  # they are additionally stored as npz files.
  if (mode == 'train' and config.dataset_size) or \
     (mode == 'crossval' and config.crossval_dataset_size):
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      dataset_size = config.dataset_size if mode == 'train' \
        else config.crossval_dataset_size
      if total <= dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    if mode == 'train':
      logger.scalar('dataset_size', total + length)
    if mode == 'crossval':
      logger.scalar('dataset_size_crossval', total + length)
  # in the following line the completed episode is additionally stored in
  # train_eps or eval_eps dictionary
  cache[str(filename)] = episode
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_episodes', len(cache))
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()


def main(config):
  logdir = pathlib.Path(config.logdir).expanduser() / f"{config.task}_{config.seed}"
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat
  config.crossval_new_dataset_every //= config.action_repeat
  config.act = getattr(tf.nn, config.act)
  print(config.task)

  run_start_time = time.time()



  if config.debug:
    tf.config.experimental_run_functions_eagerly(True)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  print('Logdir', logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  config.traindir.mkdir(parents=True, exist_ok=True)
  config.evaldir.mkdir(parents=True, exist_ok=True)
  step = count_steps(config.traindir)
  logger = tools.Logger(logdir, config.action_repeat * step, config.record_videos)

  sys.stdout = open(logdir / "logs", 'w')
  print('Logdir', logdir)
  for k, v in config.__dict__.items():
    print(f"{k:<30} {v}")

  print('Create envs.')
  if config.offline_traindir:
    directory = config.offline_traindir.format(**vars(config))
  else:
    directory = config.traindir
  train_eps = tools.load_episodes(directory, limit=config.dataset_size)
  if config.offline_evaldir:
    directory = config.offline_evaldir.format(**vars(config))
  else:
    directory = config.evaldir
  eval_eps = tools.load_episodes(directory, limit=1)
  make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, None)
  train_envs = [make('train') for _ in range(config.envs)]
  eval_envs = [make('eval') for _ in range(config.envs)]

  crossval_dir = logdir / 'crossval_eps'
  config.crossval_dir = crossval_dir
  if config.do_utd_updates:
    crossval_dir.mkdir(parents=True, exist_ok=True)
    cross_val_eps = tools.load_episodes(crossval_dir,
                                        limit=config.crossval_dataset_size)

    make_cv = lambda mode: make_env(config, logger, mode, None, None,
                                    cross_val_eps)
    # using mode=train  makes this wrapped env store trajectories
    crossval_envs = [make_cv('crossval') for _ in range(config.envs)]
  else:
    cross_val_eps = None

  acts = train_envs[0].action_space
  config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

  prefill = max(0, config.prefill - count_steps(config.traindir))
  print(f'Prefill dataset ({prefill} steps).')
  if hasattr(acts, 'discrete'):
    random_actor = tools.OneHotDist(tf.zeros_like(acts.low)[None])
  else:
    random_actor = tfd.Independent(
        tfd.Uniform(acts.low[None], acts.high[None]), 1)
  def random_agent(o, d, s):
    action = random_actor.sample()
    logprob = random_actor.log_prob(action)
    return {'action': action, 'logprob': logprob}, None
  tools.simulate(random_agent, train_envs, prefill)
  tools.simulate(random_agent, eval_envs, episodes=1)
  logger.step = config.action_repeat * count_steps(config.traindir)



  if config.do_utd_updates and not (logdir / 'variables.pkl').exists():
    if config.crossval_amount_new_data <= 3000:
      print("config.crossval_amount_new_data is set below 3000 which is needed to sample a whole batch with the "
            "current batch size, so collect at least 3000 samples initially for CV dataset")
      cv_eps_info = tools.simulate(random_agent, crossval_envs, 3000)
      cv_eps_info = cv_eps_info[0] + 3000
    else:
      cv_eps_info = tools.simulate(random_agent, crossval_envs, config.crossval_amount_new_data)
      cv_eps_info = cv_eps_info[0] + config.crossval_amount_new_data

  print('Simulate agent.')
  train_dataset = make_dataset(train_eps, config)
  eval_dataset = iter(make_dataset(eval_eps, config))
  agent = Dreamer(config, logger, train_dataset, cross_val_eps)
  if (logdir / 'variables.pkl').exists():
    agent.load(logdir / 'variables.pkl')
    print("Loaded existing agent from: ", logdir / 'variables.pkl')
    agent._should_pretrain._once = False

  if config.do_utd_updates and not (logdir / 'variables.pkl').exists() and not config.crossval_only_record_loss:
    agent._step.assign_add(cv_eps_info)
    agent._logger.step = agent._config.action_repeat * agent._step.numpy().item()
    agent.should_sample_new_cv_data._last = agent._step.numpy().item()

  state = None
  while agent._step.numpy().item() < config.steps and (time.time() - run_start_time) / 3600 < config.end_run_after:
    logger.write()
    print('Start evaluation.')
    video_pred = agent._wm.video_pred(next(eval_dataset))
    logger.video('eval_openl', video_pred)
    eval_policy = functools.partial(agent, training=False)
    tools.simulate(eval_policy, eval_envs, episodes=1)
    sys.stdout.flush()
    if agent._step.numpy().item() >= 400000:
      agent.should_sample_new_cv_data._every = int(2 * config.crossval_new_dataset_every)
    if config.do_utd_updates and\
            agent.should_sample_new_cv_data(agent._step.numpy().item()):
      cv_eps_info = tools.simulate(eval_policy, crossval_envs, config.crossval_amount_new_data)
      if not config.crossval_only_record_loss:
        agent._step.assign_add(cv_eps_info[0] + config.crossval_amount_new_data)
        agent._logger.step = agent._config.action_repeat * agent._step.numpy().item()
      num_chunks = tools.determine_num_valid_chunks_cv_dataset(
                                         agent.cross_val_eps, config.batch_length)

      print("sampled new crossval dataset with num_chunks: ", num_chunks)
    print('Start training.')
    state = tools.simulate(agent, train_envs, config.eval_every, state=state)
    agent.save(logdir / 'variables.pkl')

  if (time.time() - run_start_time) / 3600 > config.end_run_after:
    print("End Run as Timelimit has been reached")

  logger.write()
  if config.task.startswith('atari'):
    print("Start last evaluation with 100 episodes at end of Training")
    tools.simulate(eval_policy, eval_envs, episodes=100)
    df = load_jsonl(logdir / 'metrics.jsonl')[['step', 'eval_return']]
    df_last_eval = df.loc[df['eval_return'].last_valid_index()-99:df['eval_return'].last_valid_index()]
    for i in range(99):
      assert df_last_eval['step'].iloc[i] == df_last_eval['step'].iloc[i+1], \
        "Step value not the same for last 100"
    eval_mean = df_last_eval['eval_return'].mean()
    logger.scalar("last_eval_return_mean", eval_mean)
    logger.write()

  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass
  print('agent step ct', agent._step.numpy().item(), 'allowed steps', config.steps)
  print('step end condition', agent._step.numpy().item() < config.steps,
        'time end condition', (time.time() - run_start_time) / 3600 < config.end_run_after)
  print("Training ended in the normal way")
  sys.stdout.flush()
  if sys.stdout == logdir / 'logs':
    sys.stdout.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))

  main(parser.parse_args(remaining))
