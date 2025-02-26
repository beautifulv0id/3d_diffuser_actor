#!/bin/bash

# ============================================================
# REQUIRED: You must set values for these variables
# ============================================================
tasks="place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap"  # REQUIRED
dataset="$PERACT_DATA/Peract_packaged/train"  # REQUIRED
valset="$PERACT_DATA/Peract_packaged/val"  # REQUIRED

# ============================================================
# Optional: You can modify these default values
# ============================================================
# RLBench
cameras="wrist left_shoulder right_shoulder front"
image_size=256,256
max_episodes_per_task=100
instructions=$PERACT_DATA/instructions.pkl
variations=$(echo {0..199})
accumulate_grad_batches=1
gripper_loc_bounds=tasks/18_peract_tasks_location_bounds.json
gripper_loc_bounds_buffer=0.04

# Logging
val_freq=2000
base_log_dir=train_logs
exp_log_dir=$(./scripts/utils/get_log_path.sh)
name=3d_diffuser_actor_flow_matching_6D

# Training Parameters
seed=0
resume=1
eval_only=0
num_workers=1
batch_size=16
batch_size_val=4
cache_size=100
cache_size_val=0
lr=0.0001
wd=0.005
train_iters=200000
val_iters=-1
max_episode_length=5

# Dataset Augmentations
dense_interpolation=1
interpolation_length=2
rot_noise=0.0
pos_noise=0.0
pcd_noise=0.0
image_rescale=0.75,1.25

# Model Parameters
backbone=clip
embedding_dim=120
num_vis_ins_attn_layers=2
use_instruction=1
rotation_parametrization=6D
quaternion_format=wxyz
diffusion_timesteps=100
keypose_only=1
num_history=3
relative_action=0
lang_enhanced=0
fps_subsampling_factor=5

task_list=($tasks)
if [ ${#task_list[@]} -gt 1 ]; then
    task_desc="multitask"
else
    task_desc=${task_list[0]}
fi

run_log_dir=3d_diffuser_actor_flow_matching_6D_$task_desc-C$embedding_dim-B$batch_size-lr$lr-H$num_history-DT$diffusion_timesteps-RN$rot_noise-PN$pos_noise-PCDN$pcd_noise-FPS$fps_subsampling_factor


# ============================================================
# Configuration settings
# ============================================================
ngpus=$(python3 utils/count_cuda_devices.py)
CUDA_LAUNCH_BLOCKING=1

# ============================================================
# Run training command
# ============================================================
torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory_flow_matching_6D.py \
    --tasks ${tasks} \
    --dataset ${dataset} \
    --valset ${valset} \
    --cameras ${cameras} \
    --image_size ${image_size} \
    --max_episodes_per_task ${max_episodes_per_task} \
    --instructions ${instructions} \
    --variations ${variations} \
    --accumulate_grad_batches ${accumulate_grad_batches} \
    --gripper_loc_bounds ${gripper_loc_bounds} \
    --gripper_loc_bounds_buffer ${gripper_loc_bounds_buffer} \
    --val_freq ${val_freq} \
    --base_log_dir ${base_log_dir} \
    --exp_log_dir ${exp_log_dir} \
    --run_log_dir ${run_log_dir} \
    --name ${name} \
    --seed ${seed} \
    --resume ${resume} \
    --eval_only ${eval_only} \
    --num_workers ${num_workers} \
    --batch_size ${batch_size} \
    --batch_size_val ${batch_size_val} \
    --cache_size ${cache_size} \
    --cache_size_val ${cache_size_val} \
    --lr ${lr} \
    --wd ${wd} \
    --train_iters ${train_iters} \
    --val_iters ${val_iters} \
    --max_episode_length ${max_episode_length} \
    --dense_interpolation ${dense_interpolation} \
    --interpolation_length ${interpolation_length} \
    --rot_noise ${rot_noise} \
    --pos_noise ${pos_noise} \
    --pcd_noise ${pcd_noise} \
    --image_rescale ${image_rescale} \
    --backbone ${backbone} \
    --embedding_dim ${embedding_dim} \
    --num_vis_ins_attn_layers ${num_vis_ins_attn_layers} \
    --use_instruction ${use_instruction} \
    --rotation_parametrization ${rotation_parametrization} \
    --quaternion_format ${quaternion_format} \
    --diffusion_timesteps ${diffusion_timesteps} \
    --keypose_only ${keypose_only} \
    --num_history ${num_history} \
    --relative_action ${relative_action} \
    --lang_enhanced ${lang_enhanced} \
    --fps_subsampling_factor ${fps_subsampling_factor}