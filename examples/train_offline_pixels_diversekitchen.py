import os
import pickle
import numpy as np

import gym
from benchmark.domains import d4rl2
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from flax.metrics.tensorboard import SummaryWriter
from ml_collections import config_flags

from jaxrl2.evaluation import evaluate_kitchen

from jaxrl2.agents.pixel_cql import PixelCQLLearner
from jaxrl2.agents.pixel_iql import PixelIQLLearner
from jaxrl2.agents.pixel_bc import PixelBCLearner
from jaxrl2.agents import PixelIDQLLearner, PixelDDPMBCLearner
#from jaxrl2.agents.cql_encodersep_parallel import PixelCQLLearnerEncoderSepParallel

import jaxrl2.wrappers.combo_wrappers as wrappers
from jaxrl2.wrappers.frame_stack import FrameStack

import collections

from jaxrl2.data import MemoryEfficientReplayBuffer

from glob import glob

tf.config.experimental.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

# flags.DEFINE_string('env_name', 'randomized_kitchen_microwave-v1', 'Environment name.')
flags.DEFINE_string('save_dir', './results', 'Tensorboard logging dir.')
flags.DEFINE_string('project', "vd5rl", 'WandB project.')
flags.DEFINE_string('algorithm', "cql", 'Which offline RL algorithm to use.')
flags.DEFINE_string('description', "default", 'WandB project.')
flags.DEFINE_string('task', "microwave", 'Task for the kitchen env.')
flags.DEFINE_string('camera_angle', "camera2", 'Camera angle.')
flags.DEFINE_string('datadir', "microwave", 'Directory with dataset files.')
flags.DEFINE_integer('ep_length', 280, 'Episode length.')
flags.DEFINE_integer('action_repeat', 1, 'Random seed.')
flags.DEFINE_integer('replay_buffer_size', int(1e6), 'Number of transitions the (offline) replay buffer can hold.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 250,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('online_eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_gradient_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_integer('max_online_gradient_steps', int(5e5), 'Number of training steps.')
flags.DEFINE_boolean('finetune_online', True, 'Save videos during evaluation.')
flags.DEFINE_boolean('proprio', False, 'Save videos during evaluation.')
flags.DEFINE_float("take_top", None, "Take top N% trajectories.")
flags.DEFINE_float(
    "filter_threshold", None, "Take trajectories with returns above the threshold."
)
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('debug', False, 'Set to debug params (shorter).')

# config_flags.DEFINE_config_file(
#     'config',
#     './configs/offline_pixels_config.py:cql',
#     'File path to the training hyperparameter configuration.',
#     lock_config=False)


def main(_):
    from jax.lib import xla_bridge
    print('DEVICE:', xla_bridge.get_backend().platform)

    config_flags.DEFINE_config_file(
        'config',
        f'./configs/offline_pixels_diversekitchen_config.py:{FLAGS.algorithm}',
        'File path to the training hyperparameter configuration.',
        lock_config=False)

    if FLAGS.debug:
        FLAGS.project = "trash_results"
        # FLAGS.batch_size = 16
        FLAGS.max_gradient_steps = 500
        FLAGS.eval_interval = 300
        FLAGS.online_eval_interval = 300
        FLAGS.eval_episodes = 2
        FLAGS.log_interval = 200

        if FLAGS.max_online_gradient_steps > 0:
            FLAGS.max_online_gradient_steps = 500

    save_dir = os.path.join(FLAGS.save_dir, FLAGS.project, FLAGS.task, FLAGS.algorithm, FLAGS.description, f"seed_{FLAGS.seed}")
    os.makedirs(os.path.join(save_dir, "wandb"), exist_ok=True)
    group_name = f"{FLAGS.task}_{FLAGS.algorithm}_{FLAGS.description}"
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

    print('Environment Created')
    kwargs = dict(FLAGS.config.model_config)
    if kwargs.pop('cosine_decay', False):
        # kwargs['decay_steps'] = FLAGS.max_gradient_steps
        kwargs['decay_steps'] = FLAGS.max_gradient_steps + FLAGS.max_online_gradient_steps

    # assert kwargs["cnn_groups"] == 1
    print(globals()[FLAGS.config.model_constructor])
    if FLAGS.algorithm in ('idql', 'ddpm_bc'):
        agent = globals()[FLAGS.config.model_constructor].create(
        FLAGS.seed, env.observation_space, env.action_space,
        **kwargs)
    else:
        agent = globals()[FLAGS.config.model_constructor](
            FLAGS.seed, env.observation_space.sample(), env.action_space.sample(),
            **kwargs)
    print('Agent created')

    print("Loading replay buffer")
    # replay_buffer = MemoryEfficientReplayBuffer(env.observation_space, env.action_space, FLAGS.replay_buffer_size)
    # replay_buffer.seed(FLAGS.seed)
    # replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": FLAGS.batch_size, "include_pixels": False})
    # load_data(replay_buffer, FLAGS.datadir, FLAGS.task, FLAGS.ep_length, 3, FLAGS.proprio, debug=FLAGS.debug)
    replay_buffer = env.q_learning_dataset(include_pixels=False, size=FLAGS.replay_buffer_size, debug=FLAGS.debug)

    if FLAGS.take_top is not None or FLAGS.filter_threshold is not None:
        ds.filter(take_top=FLAGS.take_top, threshold=FLAGS.filter_threshold)


    print('Replay buffer loaded')

    print('Start offline training')
    tbar = tqdm.tqdm(range(1, FLAGS.max_gradient_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm)
    for i in tbar:
        tbar.set_description(f"[{FLAGS.algorithm} {FLAGS.seed}]  (offline)")
        # batch = next(replay_buffer_iterator)
        batch = replay_buffer.sample(FLAGS.batch_size)
        out = agent.update(batch)

        if isinstance(out, tuple):
            agent, update_info = out
        else:
            update_info = out

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    wandb.log({f'training/{k}': v}, step=i)
                    # print(k, v)

        if i % FLAGS.eval_interval == 0:
            eval_info = evaluate_kitchen(agent,
                                 eval_env,
                                 num_episodes=FLAGS.eval_episodes,
                                 progress_bar=False)
            for k, v in eval_info.items():
                wandb.log({f'evaluation/{k}': v}, step=i)

            wandb.log({f"replay_buffer/capacity": replay_buffer._capacity}, step=i)
            wandb.log({f"replay_buffer/size": replay_buffer._size}, step=i)
            wandb.log({f"replay_buffer/fullness": replay_buffer._size / replay_buffer._capacity}, step=i)

    eval_info = evaluate_kitchen(agent,
                         eval_env,
                         num_episodes=100,
                         progress_bar=False)
    for k, v in eval_info.items():
        wandb.log({f'evaluation/{k}': v}, step=i)

    agent.save_checkpoint(os.path.join(save_dir, "offline_checkpoints"), i, -1)

    if FLAGS.finetune_online and FLAGS.max_online_gradient_steps > 0:
        print('Start online training')
        observation, done = env.reset(), False
        tbar = tqdm.tqdm(range(1, FLAGS.max_online_gradient_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm)
        for i in tbar:
            tbar.set_description(f"[{FLAGS.algorithm} {FLAGS.seed} (online)]")

            out = agent.sample_actions(observation)
            if isinstance(out, tuple):
                action, agent = out
            else:
                action = out

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

            if done:
                for k, v in info["episode"].items():
                    decode = {"r": "return", "l": "length", "t": "time"}
                    wandb.log({f"training/{decode[k]}": v}, step=i + FLAGS.max_gradient_steps)
                observation, done = env.reset(), False

            batch = replay_buffer.sample(FLAGS.batch_size, reinsert_offline=False)
            if FLAGS.algorithm == "idql":
                out = agent.update_online(batch)
            else:
                out = agent.update(batch)

            if isinstance(out, tuple):
                agent, update_info = out
            else:
                update_info = out

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    if v.ndim == 0:
                        wandb.log({f'training/{k}': v}, step=i + FLAGS.max_gradient_steps)
                        # print(k, v)

            if i % FLAGS.online_eval_interval == 0:
                eval_info = evaluate_kitchen(agent,
                                     eval_env,
                                     num_episodes=FLAGS.eval_episodes,
                                     progress_bar=False)

                for k, v in eval_info.items():
                    if FLAGS.debug:
                        v += 1000

                    wandb.log({f'evaluation/{k}': v}, step=i + FLAGS.max_gradient_steps)

                wandb.log({f"replay_buffer/capacity": replay_buffer._capacity}, step=i + FLAGS.max_gradient_steps)
                wandb.log({f"replay_buffer/size": replay_buffer._size}, step=i + FLAGS.max_gradient_steps)
                wandb.log({f"replay_buffer/fullness": replay_buffer._size / replay_buffer._capacity}, step=i + FLAGS.max_gradient_steps)

        agent.save_checkpoint(os.path.join(save_dir, "online_checkpoints"), i + FLAGS.max_gradient_steps, -1)


def make_env(task, ep_length, action_repeat, proprio, camera_angle=None):
    suite, task = task.split('_', 1)

    if "diversekitchen" in suite:

        """
        Check what the episode length is for standard kitchen
        """

        task_set, datasets = task.split("-")
        datasets = datasets.split("+")

        if task_set == "indistribution":
            tasks_to_complete = ['microwave', 'kettle', 'switch', 'slide']
        elif task_set == "outofdistribution":
            tasks_to_complete = ['microwave', 'kettle', "bottomknob", 'switch']
        else:
            raise ValueError(f"Unsupported tasks set: \"{task_set}\".")

        # env = gym.make(task)
        env = gym.make("random_kitchen-v1",
                       tasks_to_complete=tasks_to_complete,
                       datasets=datasets,
                       framestack=3)

        print("\nenv:", env)
        print("\nenv._max_episode_steps:", env._max_episode_steps)
        print("\nenv.env.env.tasks_to_complete:", env.env.env.tasks_to_complete)
        print("\nenv.env.env._datasets_urls:", env.env.env._datasets_urls)
        print("\nenv.env.env.env._frames:", env.env.env.env._frames)
        print("\nenv.cameras:", env.cameras)
        print("\nenv.observation_space:", env.observation_space)
        # dataset = env.q_learning_dataset()
        return env
        # assert proprio
    else:
        raise ValueError(f"Unsupported environment suite: \"{suite}\".")
    return env

def load_episode(episode_file, task, tasks_list):
    with open(episode_file, 'rb') as f:
        episode = np.load(f, allow_pickle=True)

        if tasks_list is None:
            episode = {k: episode[k] for k in episode.keys() if k not in ['image_128'] and "metadata" not in k and "str" not in episode[k].dtype.name and episode[k].dtype != object}
        else:
            if "reward" in episode:
                rewards = episode["reward"]
            else:
                rewards = sum([episode[f"reward {obj}"] for obj in tasks_list])

            episode = {k: episode[k] for k in episode.keys() if k not in ['image_128'] and "metadata" not in k and "str" not in episode[k].dtype.name and episode[k].dtype != object and "init_q" not in k and "observation" not in k and "terminal" not in k and "goal" not in k}
            episode["reward"] = rewards


    # extra_image_camera_0_rgb
    # extra_image_camera_1_rgb
    # extra_image_camera_gripper_rgb

        if "standardkitchen" in task:
            keys = ["extra_image_camera_0_rgb", "extra_image_camera_1_rgb", "extra_image_camera_gripper_rgb"]
            img = np.concatenate([episode[key] for key in keys], axis=-1)
            episode["image"] = img

    return episode


if __name__ == '__main__':
    app.run(main)


"""
cd /iris/u/khatch/vd5rl/jaxrl2-irisfork/examples
conda activate jaxrlfork
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL="egl"
export KITCHEN_DATASETS=/iris/u/khatch/vd5rl/datasets/diversekitchen

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u train_offline_pixels_diversekitchen.py \
--task "diversekitchen_outofdistribution-play_data" \
--tqdm=true \
--project vd5rl_kitchen \
--algorithm cql \
--proprio=true \
--description default \
--eval_episodes 1 \
--eval_interval 200 \
--max_gradient_steps 10_000 \
--replay_buffer_size 600_000 \
--seed 0 \
--debug=true


--task "diversekitchen_indistribution-play+expert" \




XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u train_offline_pixels_diversekitchen.py \
--task "diversekitchen_indistribution-expert_demos" \
--tqdm=true \
--project dev \
--algorithm ddpm_bc \
--proprio=true \
--description proprio \
--eval_episodes 100 \
--eval_interval 50000 \
--log_interval 1000 \
--max_gradient_steps 500_000 \
--finetune_online=false \
--max_online_gradient_steps 500_000 \
--replay_buffer_size 400_000 \
--batch_size 64 \
--seed 0 \
--debug=true

"""
