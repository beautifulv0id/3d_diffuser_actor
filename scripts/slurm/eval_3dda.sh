#!/bin/bash
#SBATCH -t 5:00:00
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=eval_logs/slurm_logs/%j.out
#SBATCH -J eval_3d_diffuser_actor_peract

# Default values
exp="/home/share/3D_attn_felix/train_logs/2025.01.27/07.40.39_Actor_18Peract_100Demo_multitask/"
tasks=stack_blocks
seed="0"
data_dir="/data/peract/test"
instructions="/data/peract/instructions.pkl"
num_episodes=100
gripper_loc_bounds_file="tasks/18_peract_tasks_location_bounds.json"
max_tries=2
verbose=1
cameras="left_shoulder,right_shoulder,wrist,front"
action_dim=8
test_model="3d_diffuser_actor"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tasks) tasks="$2"; shift 2;;
    --exp) exp="$2"; shift 2;;
    --log_dir) log_dir="$2"; shift 2;;
    --seed) seed="$2"; shift 2;;
    --test_model) test_model="$2"; shift 2;;
    *) echo "Unknown parameter: $1"; exit 1;;
  esac
done

if [ -n "$log_dir"]; then
    log_dir="train_logs/$exp"
fi

checkpoint="$log_dir/best.pth"
hyper_params_file="$log_dir/hparams.json"

echo "Tasks: $tasks"

kwargs="--tasks $tasks \
        --checkpoint $checkpoint \
        --headless 1 \
        --hyper_params_file $hyper_params_file \
        --test_model $test_model \
        --cameras $cameras \
        --verbose $verbose \
        --action_dim $action_dim \
        --collision_checking 0 \
        --predict_trajectory 1 \
        --single_task_gripper_loc_bounds 0 \
        --data_dir $data_dir \
        --num_episodes $num_episodes \
        --output_file eval_logs/$exp/seed$seed/${tasks}.json \
        --instructions $instructions \
        --variations {0..60} \
        --max_tries $max_tries \
        --max_steps 25 \
        --seed $seed \
        --gripper_loc_bounds_file $gripper_loc_bounds_file \
        --gripper_loc_bounds_buffer 0.04"


bash slurm_scripts/run_python_docker.sh online_evaluation_rlbench/evaluate_policy_from_hparams.py $kwargs