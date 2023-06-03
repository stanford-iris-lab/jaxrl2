#! /usr/bin/env python
import copy
import sys
import datetime
import os
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
from jaxrl2.data.utils import get_task_id_mapping
from jaxrl2.utils.general_utils import AttrDict
from jaxrl2.agents import IQLLearner
from jaxrl2.agents.sac.sac_learner import SACLearner
from jaxrl2.agents.cql.pixel_cql_learner import PixelCQLLearner
from jaxrl2.agents.pixel_bc.pixel_bc_learner import PixelBCLearner
from jaxrl2.agents.sarsa import PixelSARSALearner
from jaxrl2.agents.cql_parallel_overall.pixel_cql_learner import PixelCQLParallelLearner
from jaxrl2.agents.cql_encodersep.pixel_cql_learner import PixelCQLLearnerEncoderSep
from jaxrl2.agents.cql_encodersep_method.pixel_cql_learner import PixelCQLLearnerEncoderSepMethod
from jaxrl2.agents.cql_encodersep_parallel.pixel_cql_learner import PixelCQLLearnerEncoderSepParallel
from jaxrl2.agents.cql_encodersep_dueling.pixel_cql_learner import PixelCQLLearnerEncoderSepDueling
from jaxrl2.wrappers.rescale_actions_wrapper import RescaleActions
from jaxrl2.wrappers.prev_action_wrapper import PrevActionStack
from jaxrl2.wrappers.state_wrapper import StateStack
from jaxrl2.wrappers.dummy_env import DummyEnv
from jaxrl2.data.replay_buffer import ReplayBuffer
from jaxrl2.wrappers.reaching_reward_wrapper import ReachingReward
from jaxrl2.utils.general_utils import add_batch_dim
from jaxrl2.kitchen_play.combo_wrappers import Kitchen, ActionRepeat, NormalizeActions, TimeLimit
from jaxrl2.data.kitchen_dataset import KitchenDataset
import collections
import jax.numpy as jnp
from flax.training import checkpoints
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict
import jax
try:
    from doodad.wrappers.easy_launch import sweep_function, save_doodad_config
except:
    print('Warning, doodad not found!')
import time
from gym.spaces import Dict

import gym
import numpy as np
from tqdm import tqdm
from absl import app, flags
from gym.spaces import Box
from ml_collections import config_flags

from jaxrl2.data import MemoryEfficientReplayBuffer, MemoryEfficientReplayBufferParallel, NaiveReplayBuffer
from jaxrl2.data.dataset import MixingReplayBuffer, MixingReplayBufferParallel, PropertyReplayBuffer

from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
from examples.configs.dataset_config_sim import *

import wandb

from jaxrl2.agents.cql.pixel_cql_learner import PixelCQLLearner
from jaxrl2.agents import PixelIQLLearner
from jaxrl2.wrappers import FrameStack, obs_latency
from examples.train_utils_sim import offline_training_loop, trajwise_alternating_training_loop, stepwise_alternating_training_loop, load_buffer
from jaxrl2.wrappers.reaching_reward_wrapper import compute_distance_reward
from jaxrl2.evaluation import evaluate
import argparse
import imp

def main(variant):
    import jax        
    variant.stochastic_evals=False
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
    outputdir = os.environ['EXP'] + '/jaxrl/' + expname
    variant.outputdir = outputdir
    print('writing to output dir ', outputdir)

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=outputdir, group_name=group_name)
    
    env = DummyEnv()

    sample_obs = add_batch_dim(env.observation_space.sample())
    sample_action = add_batch_dim(env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shapes', sample_action.shape)
    
    agent = PixelCQLLearnerEncoderSep(variant.seed, sample_obs, sample_action, **kwargs)
                                
    if variant.restore_path != '':
        agent.restore_checkpoint(variant.restore_path, reset_critic=variant.reset_critic, rescale_critic_last_layer_ratio=variant.rescale_critic_last_layer_ratio)
    
    online_replay_buffer = None
    if not variant.online_from_scratch:
        if variant.rew_func_for_target_only:
            curr_rft = variant.reward_func_type if hasattr(variant, 'reward_func_type') else 0
            variant.reward_func_type = 0
        
        if variant.rew_func_for_target_only:
            variant.reward_func_type = curr_rft

        if variant.get("online_bound_nstep_return", -1) > 0:
            print("setting nstep return off during offline phase")
            agent.online_bound_nstep_return = -1
        
        kitchenset = KitchenDataset()
        
        replay_buffer = kitchenset
        offline_training_loop(variant, agent, env, replay_buffer, None, wandb_logger, perform_control_evals=False)


def load_mult_tasks(variant, train_tasks, num_traj_cutoff=None, traj_len_cutoff=None, split_pos_neg=False, split_by_traj=False):
    if num_traj_cutoff is not None:
        num_traj_cutoff = num_traj_cutoff if num_traj_cutoff >= 0 else None
    if traj_len_cutoff is not None:
        traj_len_cutoff = traj_len_cutoff if traj_len_cutoff >=0 else None
    pos_buffer_size = neg_buffer_size = buffer_size = 0
    all_trajs = []
    for dataset_file in train_tasks:
        task_size, trajs = load_buffer(dataset_file, variant, num_traj_cutoff=num_traj_cutoff, traj_len_cutoff=traj_len_cutoff, split_pos_neg=split_pos_neg, split_by_traj=split_by_traj)
        if split_pos_neg:
            pos_size, neg_size = task_size
            pos_buffer_size += pos_size
            neg_buffer_size += neg_size
            buffer_size = (pos_buffer_size, neg_buffer_size)
        else:
            buffer_size += task_size
        all_trajs.extend(trajs)
    return all_trajs, buffer_size
    
def is_positive_sample(traj, i, variant, task_name):
    return i >= len(traj['observations']) - variant.num_final_reward_steps

def is_positive_traj(traj):
    return traj['rewards'][-1, 0] >= 1

def is_positive_traj_timestep(traj, i):
    return traj['rewards'][i, 0] >= 1

def insert_data(variant, replay_buffer, trajs, run_test=False, task_id_mapping=None, split_pos_neg=False, split_by_traj=False):
    if split_pos_neg:
        assert isinstance(replay_buffer, MixingReplayBuffer)
        pos_buffer, neg_buffer = replay_buffer.replay_buffers

    if split_by_traj:
        num_traj_pos = 0
        num_traj_neg = 0

    for traj_id, traj in enumerate(trajs):
        if variant.frame_stack == 1:
            action_queuesize = 1
        else:
            action_queuesize = variant.frame_stack - 1
        prev_actions = collections.deque(maxlen=action_queuesize)
        current_states = collections.deque(maxlen=variant.frame_stack)

        for i in range(action_queuesize):
            prev_action = np.zeros_like(traj['actions'][0])
            if run_test:
                prev_action[0] = -1
            prev_actions.append(prev_action)

        for i in range(variant.frame_stack):
            state = traj['observations'][0]['state']
            if run_test:
                state[0] = 0
            current_states.append(state)

        if split_by_traj:
            positive_traj = is_positive_traj(traj)
            if positive_traj:
                num_traj_pos += 1
            else:
                num_traj_neg += 1

        # first process rewards, masks and mc_returns
        masks = [] 
        for i in range(len(traj['observations'])):
            if variant.reward_type != 'final_one':
                reward = compute_distance_reward(traj['observations'][i]['state'][:3], TARGET_POINT, variant.reward_type)
                traj['rewards'][i] = reward
                masks.append(1.0)
            else:
                reward = traj['rewards'][i]
            
                def def_rew_func(x):
                    return x * variant.reward_scale + variant.reward_shift
                    
                if not hasattr(variant, 'reward_func_type') or variant.reward_func_type == 0:
                    rew_func = def_rew_func
                elif variant.reward_func_type == 1:
                    def rew_func(rew):
                        if rew < 0:
                            return rew * 10 # buffers where terminate when place incorrectly
                        if rew == 2:
                            return 10
                        else:
                            return -1.0
                elif variant.reward_func_type == 2:    
                    def rew_func(rew):
                        if rew == 0:
                            return -10
                        elif rew == 1:
                            return -5
                        elif rew == 2:
                            return 100
                        else:
                            assert False
                elif variant.reward_func_type == 3:    
                    def rew_func(rew):
                        if rew == 0:
                            return -20
                        elif rew == 1:
                            return -10
                        elif rew == 2:
                            return -5
                        elif rew == 3:
                            return 10
                        else:
                            assert False
                else:
                    rew_func = def_rew_func
                    
                variant.reward_func = rew_func            
                reward = rew_func(reward)
                traj['rewards'][i] = reward
                    
                if traj['rewards'][i] == 10:
                    masks.append(0.0)
                else:
                    masks.append(1.0)
        # calculate reward to go
        monte_carlo_return = calc_return_to_go(traj['rewards'].squeeze().tolist(), masks, variant.discount)
        
        if variant.get("online_bound_nstep_return", -1) > 1:
            nstep_return = calc_nstep_return(variant.online_bound_nstep_return, traj['rewards'].squeeze().tolist(), masks, variant.discount)
        else:
            nstep_return = [0] * len(masks)


# process obs, next_obs, actions and insert to buffer
        for i in range(len(traj['observations'])):
            if not split_by_traj:
                is_positive = is_positive_sample(traj, i, variant, task_name=traj['task_description'])
            else:
                is_positive = is_positive_traj_timestep(traj, i)

            obs = dict()
            if not variant.from_states:
                obs['pixels'] = traj['observations'][i]['image']
                obs['pixels'] = obs['pixels'][..., np.newaxis]
                if run_test:
                    obs['pixels'][0, 0] = i
            if variant.add_states:
                obs['state'] = np.stack(current_states, axis=-1)
            if variant.add_prev_actions:
                obs['prev_action'] = np.stack(prev_actions, axis=-1)

            action_i = traj['actions'][i]
            if run_test:
                action_i[0] = i
            prev_actions.append(action_i)

            current_state = traj['next_observations'][i]['state']
            if run_test:
                current_state[0] = i + 1
            current_states.append(current_state)  # do not delay state, therefore use i instead of i

            next_obs = dict()
            if not variant.from_states:
                next_obs['pixels'] = traj['next_observations'][i]['image']
                next_obs['pixels'] = next_obs['pixels'][..., np.newaxis]
                # if i == 0:
                #     obs['pixels'] = np.tile(obs['pixels'], [1, 1, 1, variant.frame_stack])
                if run_test:
                    next_obs['pixels'][0, 0] = i + 1
            if variant.add_states:
                next_obs['state'] = np.stack(current_states, axis=-1)
            if variant.add_prev_actions:
                next_obs['prev_action'] = np.stack(prev_actions, axis=-1)

            if task_id_mapping is not None:
                if len(task_id_mapping.keys()) > 1:
                    task_id = np.zeros((len(task_id_mapping.keys())))
                    task_id[task_id_mapping[traj['task_description']]] = 1
                    obs['task_id'] = task_id
                    next_obs['task_id'] = task_id

            if split_pos_neg:
                if positive_traj:
                    trajectory_id=pos_buffer._traj_counter
                else:
                    trajectory_id=neg_buffer._traj_counter
            else:
                trajectory_id=replay_buffer._traj_counter

            insert_dict =  dict(observations=obs,
                     actions=traj['actions'][i],
                     next_actions=traj['actions'][i+1] if len(traj['actions']) > i+1 else traj['actions'][i],
                     rewards=traj['rewards'][i],
                     next_observations=next_obs,
                     masks=masks[i],
                     dones=bool(i == len(traj['observations']) - 1),
                     trajectory_id=trajectory_id,
                     mc_returns=monte_carlo_return[i],
                     nstep_returns=nstep_return[i],
                     is_offline = 1
                     )

            if split_pos_neg:
                if positive_traj:
                    pos_buffer.insert(insert_dict)
                else:
                    neg_buffer.insert(insert_dict)
            else:
                replay_buffer.insert(insert_dict)

        if split_by_traj:
            if positive_traj:
                pos_buffer.increment_traj_counter()
            else:
                neg_buffer.increment_traj_counter()
        else:
            replay_buffer.increment_traj_counter()

    if split_by_traj:
        print('num traj pos', num_traj_pos)
        print('num traj neg', num_traj_neg)


RETURN_TO_GO_DICT = dict()

def calc_return_to_go(rewards, masks, gamma):
    global RETURN_TO_GO_DICT
    rewards_str = str(rewards) + str(masks) + str(gamma)
    if rewards_str in RETURN_TO_GO_DICT.keys():
        reward_to_go = RETURN_TO_GO_DICT[rewards_str]
    else:
        reward_to_go = [0]*len(rewards)
        prev_return = 0
        for i in range(len(rewards)):
            reward_to_go[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1]
            prev_return = reward_to_go[-i-1]
        RETURN_TO_GO_DICT[rewards_str] = reward_to_go
    return reward_to_go

NSTEP_RETURN_DICT = dict()
def calc_nstep_return(n, rewards, masks, gamma):
    global NSTEP_RETURN_DICT
    rewards_str = str(rewards) + str(masks) + str(gamma)
    if rewards_str in NSTEP_RETURN_DICT.keys():
        nstep_return = NSTEP_RETURN_DICT[rewards_str]
    else:
        nstep_return = [0]*len(rewards)
        prev_return = 0
        terminal_counts=1
        for i in range(len(rewards)):
            if i < n + terminal_counts - 1:
                nstep_return[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1]
            else:
                nstep_return[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1] - (gamma**n) * rewards[-i-1+n] * masks[-i-1]
            prev_return = nstep_return[-i-1]
            
            if i!= 0 and masks[-i-1] == 0:
                terminal_counts+=1
        NSTEP_RETURN_DICT[rewards_str] = nstep_return
    return nstep_return