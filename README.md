# jaxrl2

If you use JAXRL2 in your work, please cite this repository in publications:
```
@misc{jaxrl,
  author = {Kostrikov, Ilya},
  doi = {10.5281/zenodo.5535154},
  month = {10},
  title = {{JAXRL: Implementations of Reinforcement Learning algorithms in JAX}},
  url = {https://github.com/ikostrikov/jaxrl2},
  year = {2022},
  note = {v2}
}
```

## Installation

Run
```bash
pip install --upgrade pip

pip install -e .
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  # Note: wheels only available on linux.
```

See instructions for other versions of CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

## Tests

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= pytest tests
```

## Downloading the datasets

Download the Standard Franka Kitchen data with this command:
```bash
gsutil -m cp -r "gs://d4rl2/KITCHEN_DATA/RPL_data" .
```
Then set 
```bash
export STANDARD_KITCHEN_DATASETS=/PATH/TO/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz
```

Download the Randomized Franka Kitchen data with this command: 
```bash
gsutil -m cp -r "gs://d4rl2/KITCHEN_DATA/expert_demos" .
```
Then set 
```bash
export KITCHEN_DATASETS=/PATH/TO/datasets/randomized_kitchen
```

## Running the experiments

To replicate the results from the paper, please see the following examples:

### Standard Franka Kitchen 

The following shows how to launch IQL on the standard kitchen environment on the in-distribution evaluation tasks. See `./examples/kitchen_launch_scripts/standardkitchen` for a full list of examples for launching experiments on the standard kitchen environment.

```bash
cd ./examples
conda activate d5rl

unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL="egl"
export KITCHEN_DATASETS=/PATH/TO/datasets/randomized_kitchen
export STANDARD_KITCHEN_DATASETS=/PATH/TO/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz
export RELAY_POLICY_REPO="./benchmark/domains/relay-policy-learning/adept_envs"

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u train_offline_pixels_kitchen.py \
--task "standardkitchen_indistribution" \
--tqdm=true \
--project test_standard_kitchen \
--algorithm iql \
--proprio=true \
--description proprio \
--eval_episodes 50 \
--eval_interval 10_000 \
--online_eval_interval 10_000 \
--log_interval 1000 \
--max_gradient_steps 500_000 \
--max_online_gradient_steps 500_000 \
--replay_buffer_size 700_000 \
--batch_size 256 \
--im_size 64 \
--use_wrist_cam=false \
--camera_ids "12" \
--seed 0 
```


### Randomized Franka Kitchen 

The following shows how to launch IQL on the randomized kitchen environment on the in-distribution evaluation tasks using the expert-demo data. See `./examples/kitchen_launch_scripts/randomizedkitchen` for a full list of examples for launching experiments on the randomized kitchen environment.

```bash
cd ./examples
conda activate d5rl

unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL="egl"
export KITCHEN_DATASETS=/PATH/TO/datasets/randomized_kitchen
export STANDARD_KITCHEN_DATASETS=/PATH/TO/datasets/standard_kitchen/kitchen_demos_multitask_lexa_view_and_wrist_npz
export RELAY_POLICY_REPO="./benchmark/domains/relay-policy-learning/adept_envs"

XLA_PYTHON_CLIENT_PREALLOCATE=false python3 -u train_offline_pixels_randomizedkitchen.py \
--task "randomizedkitchen_indistribution-expert_demos" \
--tqdm=true \
--project bench_randomizedkitchen_debug3 \
--algorithm iql \
--proprio=true \
--description proprio \
--eval_episodes 100 \
--eval_interval 50000 \
--online_eval_interval 50000 \
--log_interval 1000 \
--max_gradient_steps 500_000 \
--max_online_gradient_steps 500_000 \
--replay_buffer_size 400_000 \
--batch_size 256 \
--im_size 64 \
--use_wrist_cam=false \
--camera_ids "12" \
--seed 0 
```

# Acknowledgements 

Thanks to [@evgenii-nikishin](https://github.com/evgenii-nikishin) for helping with JAX. And [@dibyaghosh](https://github.com/dibyaghosh) for helping with vmapped ensembles.