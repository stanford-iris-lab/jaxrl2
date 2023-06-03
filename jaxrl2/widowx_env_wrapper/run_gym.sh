#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate jax_recreate

curr_gpu=0
export CUDA_VISIBLE_DEVICES=$curr_gpu
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$curr_gpu

export PYTHONPATH=/nfs/kun2/users/asap7772/jaxrl2_finetuning_benchmark/:$PYTHONPATH; 
export EXP=/nfs/kun2/users/asap7772/jaxrl2_finetuning_benchmark/experiment_output
export DATA=/nfs/nfs1/

python /nfs/kun2/users/asap7772/jaxrl2_finetuning_benchmark/jaxrl2/widowx_env_wrapper/widowx_wrapped.py