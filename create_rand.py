import numpy as np
import argparse
import os

path = '/nfs/kun2/users/asap7772/binsort_bridge/04_26_collect_multitask/actionnoise0.0_binnoise0.0_policysorting_sparse0/train/out.npy'
output_path = '/nfs/kun2/users/asap7772/binsort_bridge/test/actionnoise0.0_binnoise0.0_policysorting_sparse0/train/'
num_trials = 3

trajs = np.load(path, allow_pickle=True)
sampled = np.random.choice(trajs, num_trials, replace=False)

os.makedirs(output_path, exist_ok=True)
np.save(output_path + '/out.npy', sampled)
