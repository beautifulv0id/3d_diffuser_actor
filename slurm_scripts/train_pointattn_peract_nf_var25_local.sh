#!/bin/bash

config_name=PointAttn_18Peract_100Demo_multitask
dataset=/home/share/3D_attn_felix/Peract_packaged/train
dataset=/media/funk/INTENSO/300_RL_BENCH/data/peract_new/train
valset=/home/share/3D_attn_felix/Peract_packaged/val
valset=/media/funk/INTENSO/300_RL_BENCH/data/peract_new/val

instruction_path=/home/share/3D_attn_felix/peract/instructions.pkl
instruction_path=/media/funk/INTENSO/300_RL_BENCH/peract_from_cluster/instructions.pkl
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json

lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=3
diffusion_timesteps=100
B=1
C=120
ngpus=1
quaternion_format=xyzw

CUDA_LAUNCH_BLOCKING=1 WANDB_PROJECT=3d_diffuser_actor_debug torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_pointattn_nf25_3_1.py \
    --tasks insert_onto_square_peg \
    --dataset $dataset \
    --valset $valset \
    --instructions $instruction_path \
    --gripper_loc_bounds $gripper_loc_bounds_file \
    --num_workers 1 \
    --train_iters 600000 \
    --embedding_dim $C \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq 2500 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 2000 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations {0..199} \
    --lr $lr\
    --num_history $num_history \
    --cameras left_shoulder right_shoulder wrist front\
    --max_episodes_per_task -1 \
    --quaternion_format $quaternion_format \
    --run_log_dir diffusion_multitask-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps
