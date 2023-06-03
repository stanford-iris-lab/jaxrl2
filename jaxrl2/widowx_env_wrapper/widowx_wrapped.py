import sys

sys.path.append('/nfs/kun2/users/asap7772/finetuning_benchmark')
sys.path.append('/nfs/kun2/users/asap7772/finetuning_benchmark/data_collection')

import numpy as np
from dm_control.composer.environment import Environment
from policies.binsort_policy import BinSortPolicy
from policies.pickplace_policy import PickPlacePolicy
from policies.base_policy import ZeroPolicy, RandomPolicy

from wrapper import DataSaver

from benchmark.domains.widowx.tasks import Drawer, PickAndPlaceTask, BinsortTask
import os
import datetime
import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import jaxrl2.wrappers.dmc2gym
from gym.spaces import Box, Dict
import gym

def get_binsort_env(category=['shoe', 'toy'], num_objects=[1, 1], task_id=[3,1], sparse_reward=0):
    task = BinsortTask(category, num_objects, task_id, sparse_reward=sparse_reward)
    env = Environment(task=task, strip_singleton_obs_buffer_dim=True)
    return env

from jaxrl2.widowx_env_wrapper.dmc2gym import DMC2GYM
from jaxrl2.data.eps_transition_dataset import default_obs_remapping, remap_dict

class ObservationWrapper(gym.core.Wrapper):
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError

class BinsortObsWrapper(ObservationWrapper):
    def __init__(self, env, task_id=[3,1]):
        super().__init__(env)
        self.observation_space = Dict({
            'pixels': Box(low=0, high=255, shape=(128, 128, 3, 1), dtype=np.uint8),
            'state': Box(low=-np.inf, high=np.inf, shape=(17, 1), dtype=np.float32),
        })
        self.one_hot_task_id = np.zeros((2,5), dtype=np.float32)
        self.one_hot_task_id[0, task_id[0]] = 1
        self.one_hot_task_id[1, task_id[1]] = 1
        self.one_hot_task_id = self.one_hot_task_id.flatten()
    
    def observation(self, obs):
        remapping = default_obs_remapping()
        obs['task_id'] = self.one_hot_task_id
        obs = remap_dict(obs, remapping)
        
        for k, v in obs.items():
            obs[k] = v[..., None]
        return obs
    
def get_gym_binsort(task_id=[3,1]):
    dm_env = get_binsort_env(task_id=task_id)
    gym_env = DMC2GYM(dm_env)
    gym_env = BinsortObsWrapper(gym_env, task_id=task_id)
    return gym_env
    
if __name__ == '__main__':
    gym_env = get_gym_binsort()
    
    obs = gym_env.reset()
    for i in range(100):
        gym_env.render()
        obs, reward, done, info = gym_env.step(np.random.randn(7))
        print(f"obs: {obs}")
        print(f"reward: {reward}, done: {done}")
        if done:
            break
