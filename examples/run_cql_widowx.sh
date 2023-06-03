#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate jax_recreate

proj_name=0603_cql_widowx
# proj_name=test
tpu_id=0
tpu_port=$(( $tpu_id+8820 ))
export PYTHONPATH=/nfs/kun2/users/asap7772/jaxrl2_finetuning_benchmark/:$PYTHONPATH; 
export EXP=/nfs/kun2/users/asap7772/jaxrl2_finetuning_benchmark/experiment_output
export DATA=/nfs/nfs1/
export MUJOCO_GL=osmesa

seed=0
cql_alpha=5
dry_run=0

total_runs=0
max_runs=14
gpu_id=0
which_devices=(1 2 3 4 5 6 7)
alphas=(0.1 1 5 10 20)
datasets=(sorting pickplace sorting_pickplace)

for alpha in ${alphas[@]}; do
for dataset in ${datasets[@]}; do

prefix=${proj_name}_${dataset}_cql_alpha_${alpha}_dataset_${dataset}_seed_${seed}
which_gpu=${which_devices[$gpu_id]}
export CUDA_VISIBLE_DEVICES=$which_gpu
echo "Running on GPU $which_gpu"

command="XLA_PYTHON_CLIENT_PREALLOCATE=false python3 examples/launch_train_widowx_cql.py \
--prefix $prefix \
--cql_alpha $alpha \
--wandb_project ${proj_name} \
--batch_size 64 \
--encoder_type small  \
--policy_encoder_type small \
--actor_lr 0.0001 \
--critic_lr 0.0003 \
--dataset $dataset \
--seed $seed \
--offline_finetuning_start -1 \
--online_start 10000000000000 \
--max_steps  10000000000000 \
--eval_interval 5000 \
--eval_episodes 2 \
--checkpoint_interval 5000 \
--tpu_port $tpu_port \
--multi_grad_step 5"

echo $command

if [ $dry_run -eq 0 ]; then
    eval $command &
    sleep 10
fi

gpu_id=$(( $gpu_id+1 ))
if [ $gpu_id -eq 7 ]; then
    gpu_id=0
fi

total_runs=$(( $total_runs+1 ))
if [ $total_runs -eq $max_runs ]; then
    exit
fi

done
done