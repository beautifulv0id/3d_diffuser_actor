#!/bin/bash

# ============================================================
# REQUIRED: You must set values for these variables
# ============================================================
tasks="place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap"  # REQUIRED
dataset="/home/share/3D_attn_felix/Peract_packaged/train/"  # REQUIRED
valset="/home/share/3D_attn_felix/Peract_packaged/val/"  # REQUIRED

# ============================================================
# Optional: You can modify these default values
# ============================================================
# RLBench
cameras="wrist left_shoulder right_shoulder front"
max_episodes_per_task=100
instructions=/home/share/3D_attn_felix/rlbench_instructions/instructions.pkl
variations=$(seq 0 199)
accumulate_grad_batches=1
gripper_loc_bounds=tasks/18_peract_tasks_location_bounds.json
gripper_loc_bounds_buffer=0.04

# Logging
val_freq=500
base_log_dir=/home/stud_herrmann/3d_diffuser_actor/train_logs
exp_log_dir=$(./scripts/utils/get_log_path.sh)
name=pointattn_self_attn

# Training Parameters
seed=0
#checkpoint="" # Set this value to resume training
resume=1
eval_only=0
num_workers=1
batch_size=16
batch_size_val=4
cache_size=100
cache_size_val=100
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
backbone=clip
embedding_dim=120
quaternion_format=wxyz
diffusion_timesteps=100
keypose_only=1
num_history=1
relative_action=0
fps_subsampling_factor=1
scaling_factor=3.0
use_normals=0
rot_factor=1.0
gripper_depth=2
decoder_depth=4
decoder_dropout=0.2
distance_scale=1.0
use_adaln=0
gripper_history_as_points=0
feature_type=sinusoid
use_center_distance="1"
use_center_projection="1"
use_vector_projection="1"
add_center=1

task_list=($tasks)
if [ ${#task_list[@]} -gt 1 ]; then
    task_desc="multitask"
else
    task_desc=${task_list[0]}
fi

run_log_dir=pointattn_self_attn_$task_desc-C$embedding_dim-B$batch_size-lr$lr-H$num_history-DT$diffusion_timesteps-RN$rot_noise-PN$pos_noise-PCDN$pcd_noise-FPS$fps_subsampling_factor-UCD$use_center_distance-UCP$use_center_projection-UVP$use_vector_projection-AC$add_center-FR$feature_res-FT$feature_type-DS$distance_scale-ADALN$use_adaln


# ============================================================
# Configuration settings
# ============================================================
ngpus=$(python3 utils/count_cuda_devices.py)
CUDA_LAUNCH_BLOCKING=1

# ============================================================
# Run training command
# ============================================================
torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_pointattn_self_attn.py \
    --tasks ${tasks} \
    --valset ${valset} \
    --dataset ${dataset} \
    --cameras ${cameras} \
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
    --feature_res ${feature_res} \
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
#    --checkpoint $checkpoint # Set this value to resume training