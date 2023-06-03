from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import defaultdict, OrderedDict, deque
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict
import numpy as np
import os
import gc
import jax
from jaxrl2.data.dataset import Dataset
import tqdm

RETURN_TO_GO_DICT = dict()
def calc_return_to_go(rewards, masks=None, gamma=0.99):
    if masks is None:
        masks = rewards
    global RETURN_TO_GO_DICT
    rewards_str = str(rewards) + str(masks) + str(gamma)
    if rewards_str in RETURN_TO_GO_DICT.keys():
        reward_to_go = RETURN_TO_GO_DICT[rewards_str]
    else:
        reward_to_go = [0]*len(rewards)
        prev_return = rewards[-1]/(1-gamma)
        for i in range(len(rewards)):
            reward_to_go[-i-1] = rewards[-i-1] + gamma * prev_return * masks[-i-1]
            prev_return = reward_to_go[-i-1]
        RETURN_TO_GO_DICT[rewards_str] = reward_to_go
    return reward_to_go

def is_dict_like(x):
    return isinstance(x, dict) or isinstance(x, FrozenDict) or isinstance(x, defaultdict) or isinstance(x, OrderedDict)
    
    
def default_remapping():
    return {}

def default_obs_remapping():
    mapping_obs = OrderedDict(
        end_effector_pos = 'state', 
        right_finger_qpos = 'state', 
        right_finger_qvel = 'state', 
        left_finger_qpos = 'state', 
        left_finger_qvel = 'state', 
        pixels = 'pixels', 
        task_id = 'state'
    )
    
    return mapping_obs

def remap_dict(d, remapping):
    new_d = {}
    for k in d.keys():
        remapped_k = remapping.get(k, k)
        if remapped_k in new_d.keys():
            new_d[remapped_k] = np.concatenate([new_d[remapped_k], d[k]], axis=-1)
        else:
            new_d[remapped_k] = np.array(d[k])
    return new_d

def npify_dict(d):
    for k in d.keys():
        if is_dict_like(d[k]):
            d[k] = npify_dict(d[k])
        else:
            d[k] = np.array(d[k])
    return d

def append_dicts(dict1, dict2):
    assert set(dict1.keys()) == set(dict2.keys()), f"Keys don't match: {dict1.keys()} vs {dict2.keys()}"
    for k in dict1.keys():
        if is_dict_like(dict1[k]):
            dict1[k] = append_dicts(dict1[k], dict2[k])
        else:
            dict1[k] = np.concatenate([dict1[k], dict2[k]], axis=0)
    return dict1
    
def append_nested_dict(concat_dict, addition, remapping={}, obs_remapping={}):
    # convert addition to correct format
    new_format_dict = {}
    for k in addition.keys():
        for i in range(len(addition[k])):
            if is_dict_like(addition[k][i]):
                for subk in addition[k][i].keys():
                    if k not in new_format_dict.keys():
                        new_format_dict[k] = {}
                    if subk not in new_format_dict[k].keys():
                        new_format_dict[k][subk] = []
                    new_format_dict[k][subk].append(addition[k][i][subk])
            else:
                if k not in new_format_dict.keys():
                    new_format_dict[k] = []
                new_format_dict[k].append(addition[k][i])
    new_format_dict = npify_dict(new_format_dict)
    # now remap and append
    new_format_dict = remap_dict(new_format_dict, remapping)
    new_format_dict['observations'] = remap_dict(new_format_dict['observations'].item(), obs_remapping)
    new_format_dict['next_observations'] = remap_dict(new_format_dict['next_observations'].item(), obs_remapping)
    
    if not concat_dict:
        return new_format_dict
    return append_dicts(concat_dict, new_format_dict)

class EpisodicTransitionDataset(Dataset):    
    def __init__(self, paths: Any, remapping=default_remapping(), obs_remapping=default_obs_remapping()):
        if isinstance(paths, str):
            paths = [paths]
        assert isinstance(paths, list)
        
        self._paths = paths
        self.episodes = []
        self.episode_as_dict = None

        self.episodes_lens = []
        for path in paths:
            assert os.path.exists(path), f'Path {path} does not exist'
            print('Loading data from', path)
            
            data = np.load(path, allow_pickle=True).tolist()
            
            self.episodes.extend(data)
            
            succ = []
            for i in tqdm.tqdm(range(len(data))):
                rews = np.array(data[i]['rewards'])
                data[i]['mc_returns'] = calc_return_to_go(rews)
                
                succ.append(rews.any())
                self.episodes_lens.append(len(rews))
                
                self.episode_as_dict = append_nested_dict(self.episode_as_dict, data[i], remapping=remapping, obs_remapping=obs_remapping)
            
            print('Success rate:', np.mean(succ))
            gc.collect()
        
        self.episodes_lens = np.array(self.episodes_lens)
        self.episodes = np.array(self.episodes)
        super().__init__(self.episode_as_dict)
        
        print('Total number of episodes:', len(self.episodes))
        
    def get_iterator(self,
                    batch_size: int,
                    keys: Optional[Iterable[str]] = None,
                    indx: Optional[np.ndarray] = None,
                    queue_size: int = 2):
    # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
    # queue_size = 2 should be ok for one GPU.

        queue = deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size, keys, indx)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
                    

def main():
    path = '/nfs/kun2/users/asap7772/binsort_bridge/test/actionnoise0.0_binnoise0.0_policysorting_sparse0/train/out.npy'
    dataset = EpisodicTransitionDataset(path)
    batch = dataset.sample(13)
    breakpoint()
    
if __name__ == '__main__':
    main()