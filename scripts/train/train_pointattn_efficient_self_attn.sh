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
max_episodes_per_task=100
instructions=/workspace/data/instructions.pkl
variations=$(echo {0..199})
accumulate_grad_batches=1

# Logging
val_freq=2000
base_log_dir=train_logs
exp_log_dir=$(./scripts/utils/get_log_path.sh)
name=pointattn_efficient_self_attn

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
feature_res=res3
max_workspace_points=tasks/max_workspace_points.json
backbone=clip
embedding_dim=120
quaternion_format=wxyz
diffusion_timesteps=100
keypose_only=1
num_history=3
relative_action=0
fps_subsampling_factor=1
scaling_factor=3.0
use_normals=0
rot_factor=1.0
gripper_depth=2
decoder_depth=4
decoder_dropout=0.0
distance_scale=1.0
use_adaln=1
gripper_history_as_points=1
feature_type=sinusoid
use_center_distance="1"
use_center_projection="1"
use_vector_projection="1"
add_center=1
point_embedding_dim=120
crop_workspace=0

task_list=($tasks)
if [ ${#task_list[@]} -gt 1 ]; then
    task_desc="multitask"
else
    task_desc=${task_list[0]}
fi

run_log_dir=pointattn_efficient_self_attn_$task_desc-C$embedding_dim-B$batch_size-lr$lr-H$num_history-DT$diffusion_timesteps-RN$rot_noise-PN$pos_noise-PCDN$pcd_noise-FPS$fps_subsampling_factor-UCD$use_center_distance-UCP$use_center_projection-UVP$use_vector_projection-AC$add_center-FR$feature_res-FT$feature_type-DS$distance_scale-ADALN$use_adaln


# ============================================================
# Configuration settings
# ============================================================
ngpus=$(python3 utils/count_cuda_devices.py)
CUDA_LAUNCH_BLOCKING=1

# ============================================================
# Run training command
# ============================================================
torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_pointattn_efficient_self_attn.py \
    --valset ${valset} \
    --tasks ${tasks} \
    --dataset ${dataset} \
    --cameras ${cameras} \
    --max_episodes_per_task ${max_episodes_per_task} \
    --instructions ${instructions} \
    --variations ${variations} \
    --accumulate_grad_batches ${accumulate_grad_batches} \
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
    --feature_res ${feature_res} \
    --max_workspace_points ${max_workspace_points} \
    --backbone ${backbone} \
    --embedding_dim ${embedding_dim} \
    --quaternion_format ${quaternion_format} \
    --diffusion_timesteps ${diffusion_timesteps} \
    --keypose_only ${keypose_only} \
    --num_history ${num_history} \
    --relative_action ${relative_action} \
    --fps_subsampling_factor ${fps_subsampling_factor} \
    --scaling_factor ${scaling_factor} \
    --use_normals ${use_normals} \
    --rot_factor ${rot_factor} \
    --gripper_depth ${gripper_depth} \
    --decoder_depth ${decoder_depth} \
    --decoder_dropout ${decoder_dropout} \
    --distance_scale ${distance_scale} \
    --use_adaln ${use_adaln} \
    --gripper_history_as_points ${gripper_history_as_points} \
    --feature_type ${feature_type} \
    --use_center_distance ${use_center_distance} \
    --use_center_projection ${use_center_projection} \
    --use_vector_projection ${use_vector_projection} \
    --add_center ${add_center} \
    --point_embedding_dim ${point_embedding_dim} \
    --crop_workspace ${crop_workspace}