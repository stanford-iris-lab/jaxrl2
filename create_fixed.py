import numpy as np
import argparse
import os
from collections import defaultdict
import tqdm
import glob
import gc

glob_path = '/nfs/kun2/users/asap7772/binsort_bridge/*/*/train/out.npy'
paths = glob.glob(glob_path)
replace_name = 'binsort_bridge_sorted_20'
output_paths = [os.path.dirname(path).replace('binsort_bridge', 'binsort_bridge_sorted') + '/' for path in paths]
num_trials = 20

with tqdm.tqdm(total=len(paths)) as pbar:
    for path, output_path in zip(paths, output_paths):
        trajs = np.load(path, allow_pickle=True)
        which_idxs = defaultdict(lambda : {2:[], 1:[], 0:[]})

        num_2 = 0
        num_1 = 0
        num_0 = 0

        for i in tqdm.tqdm(range(len(trajs))):
            zero_hot_tid = trajs[i]['observations'][0]['task_id'].reshape(2, -1)
            tid = tuple(np.argmax(zero_hot_tid, axis=-1).tolist())
            
            last_reward = trajs[i]['rewards'][-1]
            
            if last_reward == 2:
                print(f'last reward is 2 for tid: {tid}')
                num_2 += 1
            elif last_reward == 1:
                num_1 += 1
            elif last_reward == 0:
                num_0 += 1
            else:
                raise ValueError(f'last reward is not 0, 1, or 2: {last_reward}')
            
            which_idxs[tid][last_reward].append(i)
            
        print(f'num_2: {num_2}')
        print(f'num_1: {num_1}')
        print(f'num_0: {num_0}')

        which_idxs = {k: {k2: np.array(v2) for k2, v2 in v.items()} for k, v in which_idxs.items()}
        which_idxs_ordered = {k: np.concatenate([v[2], v[1], v[0]]) for k, v in which_idxs.items()}

        which_idxs_ordered = {k: v[:min(len(v), num_trials)] for k, v in which_idxs_ordered.items()}
        all_idx_chosen = np.concatenate(list(which_idxs_ordered.values())).astype(int)
        print(f'len(all_idx_chosen): {len(all_idx_chosen)}')

        sampled = trajs[all_idx_chosen]

        os.makedirs(output_path, exist_ok=True)
        np.save(output_path + '/out.npy', sampled)
        
        gc.collect()
        pbar.update(1)


# sampled = np.random.choice(trajs, num_trials, replace=False)

# os.makedirs(output_path, exist_ok=True)
# np.save(output_path + '/out.npy', sampled)
