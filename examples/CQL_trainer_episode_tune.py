import os
import pickle
import numpy as np

import gym
import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

from jaxrl2.agents.pixel_cql import PixelCQLLearner
from jaxrl2.evaluation import evaluate_kitchen

import combo_wrappers as wrappers
from d4rl2.wrappers.frame_stack import FrameStack

import collections

from jaxrl2.data import MemoryEfficientReplayBuffer

from glob import glob

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

# flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './results', 'Tensorboard logging dir.')
flags.DEFINE_string('project', "CQL_clean_v7", 'WandB project.')
flags.DEFINE_string('description', "default", 'WandB project.')
flags.DEFINE_string('task', "microwave", 'WandB project.')
flags.DEFINE_string('camera_angle', "camera2", 'WandB project.')
flags.DEFINE_string('datadir', "microwave", 'WandB project.')
flags.DEFINE_integer('ep_length', 280, 'Random seed.')
flags.DEFINE_integer('action_repeat', 1, 'Random seed.')
flags.DEFINE_integer('num_finetuning_steps', 50, 'Random seed.')
flags.DEFINE_integer('replay_buffer_size', int(1e6), 'Random seed.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 250,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_offline_gradient_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_integer('max_online_gradient_steps', 20_000, 'Number of training steps.')
flags.DEFINE_boolean('proprio', False, 'Save videos during evaluation.')


flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('debug', False, 'Save videos during evaluation.')
config_flags.DEFINE_config_file(
    'config',
    './configs/offline_pixels_config.py:cql',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def make_env(task, ep_length, action_repeat, proprio, camera_angle=None):
    suite, task = task.split('_', 1)

    if "kitchen" in suite:
        assert not proprio
        assert action_repeat == 1
        tasks_list = task.split("+")
        env = wrappers.Kitchen(task=tasks_list, size=(64, 64), proprio=proprio)
        env = wrappers.ActionRepeat(env, action_repeat)
        env = wrappers.NormalizeActions(env)
        env = wrappers.TimeLimit(env, ep_length)
        env = FrameStack(env, num_stack=3)
    elif "adroithand" in suite:
        assert proprio
        assert action_repeat == 1
        if "human" in task and "cloned" in task:
            task = task.replace("-cloned", "")

        env = wrappers.AdroitHand(task, 64, 64, proprio=proprio, camera_angle=camera_angle)
        env = wrappers.ActionRepeat(env, action_repeat)
        env = wrappers.NormalizeActions(env)
        env = wrappers.TimeLimit(env, ep_length)
        env = FrameStack(env, num_stack=3)
    elif "metaworld" in suite:
        assert not proprio
        assert action_repeat == 2
        env = wrappers.MetaWorldEnv(name=task, action_repeat=action_repeat, size=(64, 64))
        # env = wrappers.ActionRepeat(env, 2)
        env = wrappers.NormalizeActions(env)
        env = wrappers.TimeLimit(env, ep_length)
        env = FrameStack(env, num_stack=3)
    else:
        raise ValueError(f"Unsupported environment suite: \"{suite}\".")
    return env


def main(_):
    from jax.lib import xla_bridge
    print('DEVICE:', xla_bridge.get_backend().platform)

    if FLAGS.debug:
        FLAGS.project = "trash_results"
        FLAGS.max_offline_gradient_steps = 500
        FLAGS.max_online_gradient_steps = 500
        FLAGS.batch_size = 32

    save_dir = os.path.join(FLAGS.save_dir, FLAGS.project, FLAGS.task, "CQL", FLAGS.description, f"seed_{FLAGS.seed}")
    os.makedirs(os.path.join(save_dir, "wandb"), exist_ok=True)
    group_name = f"{FLAGS.task}_CQL_{FLAGS.description}"
    name = f"seed_{FLAGS.seed}"

    wandb.init(project=FLAGS.project,
               dir=os.path.join(save_dir, "wandb"),
               id=group_name + "-" + name,
               group=group_name,
               save_code=True,
               name=name,
               resume=None,
               entity="iris_intel")

    wandb.config.update(FLAGS)

    env = make_env(FLAGS.task, FLAGS.ep_length, FLAGS.action_repeat, FLAGS.proprio, FLAGS.camera_angle)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    eval_env = make_env(FLAGS.task, FLAGS.ep_length, FLAGS.action_repeat, FLAGS.proprio, FLAGS.camera_angle)

    max_online_env_steps = (FLAGS.max_online_gradient_steps // FLAGS.num_finetuning_steps) * FLAGS.ep_length
    print("max_online_env_steps:", max_online_env_steps)

    print('Environment Created')
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = FLAGS.max_offline_gradient_steps + FLAGS.max_online_gradient_steps

    assert kwargs["cnn_groups"] == 1

    print(globals()[FLAGS.config.model_constructor])
    agent = globals()[FLAGS.config.model_constructor](
        FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
        **kwargs)
    print('Agent created')

    print("Loading replay buffer")
    replay_buffer = MemoryEfficientReplayBuffer(env.observation_space, env.action_space, FLAGS.replay_buffer_size)
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False})
    load_data(replay_buffer, FLAGS.datadir, FLAGS.task, max_online_env_steps, FLAGS.ep_length, 3, FLAGS.proprio, debug=FLAGS.debug)
    print('Replay buffer loaded')

    print('Start offline training')
    tbar = tqdm.tqdm(range(1, FLAGS.max_offline_gradient_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm)
    for i in tbar:
        tbar.set_description(f"[CQL {FLAGS.seed}]")
        batch = next(replay_buffer_iterator)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb.log({f'training/{k}': v}, step=i)
                    print(k, v)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate_kitchen(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes,
                                 progress_bar=False)
            for k, v in eval_info.items():
                wandb.log({f'evaluation/{k}': v}, step=i)

    agent.save_checkpoint(os.path.join(save_dir, "offline_checkpoints"), i, -1)

    print('Start online training')
    tbar = tqdm.tqdm(range(1, max_online_env_steps // FLAGS.ep_length + 1), smoothing=0.1, disable=not FLAGS.tqdm)
    # tbar = tqdm.tqdm(range(1, FLAGS.max_online_env_steps // 50 + 1), smoothing=0.1, disable=not FLAGS.tqdm)
    for ep_no in tbar:
        tbar.set_description(f"[CQL {FLAGS.seed}] Ep {ep_no}/{max_online_env_steps // FLAGS.ep_length + 1}, i {i}/{FLAGS.max_offline_gradient_steps + FLAGS.max_online_gradient_steps}")
        observation, done = env.reset(), False
        while not done:
            action = agent.sample_actions(observation)
            # next_observation, reward, done, info = env.step(action)
            env_step = env.step(action)
            if len(env_step) == 4:
                next_observation, reward, done, info = env_step
            elif len(env_step) == 5:
                next_observation, reward, done, _, info = env_step
            else:
                raise ValueError(f"env_step should be length 4 or 5 but is length {len(env_step)}")

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(
                dict(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation,
                )
            )
            observation = next_observation

        for k, v in info["episode"].items():
            decode = {"r": "return", "l": "length", "t": "time"}
            wandb.log({f"training/{decode[k]}": v}, step=i)


        for grad_idx in range(FLAGS.num_finetuning_steps):
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            assert i >= FLAGS.max_offline_gradient_steps
            i += 1 # This i comes from the previous for loop

        for k, v in update_info.items():
            if v.ndim == 0:
                wandb.log({f'training/{k}': v}, step=i)
                print(k, v)

        eval_info = evaluate_kitchen(agent,
                             eval_env,
                             num_episodes=FLAGS.eval_episodes,
                             progress_bar=False) ###===### ###---###

        for k, v in eval_info.items():
            if FLAGS.debug:
                v += 1000

            wandb.log({f'evaluation/{k}': v}, step=i)

    agent.save_checkpoint(os.path.join(save_dir, "online_checkpoints"), i, -1)


def load_episode(episode_file, task):
    with open(episode_file, 'rb') as f:
        episode = np.load(f, allow_pickle=True)

        if task is None:
            episode = {k: episode[k] for k in episode.keys() if k not in ['image_128'] and "metadata" not in k and "str" not in episode[k].dtype.name and episode[k].dtype != object}
        else:
            if "reward" in episode:
                rewards = episode["reward"]
            else:
                rewards = sum([episode[f"reward {obj}"] for obj in task])

            episode = {k: episode[k] for k in episode.keys() if k not in ['image_128'] and "metadata" not in k and "str" not in episode[k].dtype.name and episode[k].dtype != object and "init_q" not in k and "observation" not in k and "terminal" not in k and "goal" not in k}
            episode["reward"] = rewards
    return episode


def load_data(replay_buffer, offline_dataset_path, task, max_online_env_steps, ep_length, num_stack, proprio, debug=False):
    if "kitchen" in task:
        tasks_list = task.split("_")[-1].split("+")
    else:
        tasks_list = None

    episode_files = glob(os.path.join(offline_dataset_path, '**', '*.npz'), recursive=True)
    total_transitions = 0

    for episode_file in tqdm.tqdm(episode_files, total=len(episode_files), desc="Loading offline data"):
        episode = load_episode(episode_file, tasks_list)

        # observation, done = env.reset(), False
        frames = collections.deque(maxlen=num_stack)
        for _ in range(num_stack):
            frames.append(episode["image"][0])

        observation = dict(pixels=np.stack(frames, axis=-1))
        if proprio:
            observation["states"] = episode["proprio"][0]
        done = False

        i = 1
        while not done:
            # action = agent.sample_actions(observation)
            action = episode["action"][i]

            # next_observation, reward, done, info = env.step(action)
            frames.append(episode["image"][i])
            next_observation = dict(pixels=np.stack(frames, axis=-1))
            if proprio:
                next_observation["states"] = episode["proprio"][i]
            reward = episode["reward"][i]
            done = i >= episode["image"].shape[0] - 1
            # print(f"i: {i}, done: {done}")
            info = {}

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0
            replay_buffer.insert(
                dict(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation,
                )
            )
            observation = next_observation
            total_transitions += 1
            i += 1

            if debug and total_transitions > 5000:
                return

    print(f"Loaded {len(episode_files)} episodes and {total_transitions} total transitions.")
    print(f"replay_buffer capacity {replay_buffer._capacity}, replay_buffer size {replay_buffer._size}, total online and offline steps {total_transitions + (max_online_env_steps / 50) * ep_length}.")
    assert replay_buffer._capacity > total_transitions + (max_online_env_steps / 50) * ep_length


if __name__ == '__main__':
    app.run(main)
