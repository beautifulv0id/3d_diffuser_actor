#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=train_logs/slurm_logs/%j_train.out
#SBATCH -J actor
#SBATCH -C 'rtx3090|a5000|a6000'

config_name=Actor_18Peract_100Demo_multitask
main_dir=$(./scripts/get_log_path.sh $config_name)

dataset=/home/share/3D_attn_felix/Peract_packaged/train
valset=/home/share/3D_attn_felix/Peract_packaged/val

lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=3
diffusion_timesteps=100
B=2
C=120
ngpus=$(python3 scripts/helper/count_cuda_devices.py)
quaternion_format=xyzw

CUDA_LAUNCH_BLOCKING=1 WANDB_PROJECT=3d_diffuser_actor_debug torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory.py \
    --tasks sweep_to_dustpan_of_size \
    --dataset $dataset \
    --valset $valset \
    --instructions /home/share/3D_attn_felix/rlbench_instructions/instructions.pkl \
    --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json \
    --num_workers 1 \
    --train_iters 600000 \
    --embedding_dim $C \
    --use_instruction 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq 4000 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 600 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations {0..199} \
    --lr $lr\
    --num_history $num_history \
    --cameras left_shoulder right_shoulder wrist front\
    --max_episodes_per_task -1 \
    --quaternion_format $quaternion_format \
    --run_log_dir diffusion_multitask-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps
