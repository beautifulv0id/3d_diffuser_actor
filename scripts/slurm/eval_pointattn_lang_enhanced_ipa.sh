#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --output=eval_logs/slurm_logs/%A_pointattn_lang_enhanced_ipa/%a.out
#SBATCH -J eval_pointattn_lang_enhanced_ipa

checkpoint_dir=$1
seeds=$(echo {0..3})
tasks=(place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap)
base_log_dir=$(echo $checkpoint_dir | cut -d"/" -f1)
exp_log_dir=$(echo $checkpoint_dir | cut -d"/" -f2)/$(echo $checkpoint_dir | cut -d"/" -f3)
run_log_dir=$(echo $checkpoint_dir | cut -d"/" -f4)
test_model=pointattn_lang_enhanced_ipa
variations=$(echo {0..60})
checkpoint=$checkpoint_dir/last.pth
hparams=$checkpoint_dir/hparams.json
instructions=/workspace/data/instructions.pkl
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
cameras="left_shoulder,right_shoulder,wrist,front"
single_task_gripper_loc_bounds=0
data_dir=/workspace/data/test/
num_episodes=100
max_tries=2

seed_array=($seeds)
seed=${seed_array[$SLURM_ARRAY_TASK_ID % ${#seed_array[@]}]}

task_index=$(($SLURM_ARRAY_TASK_ID / ${#seed_array[@]}))
task=${tasks[$task_index]}

echo "Starting docker container"
id=$(docker run -dt \
   -e WANDB_API_KEY=$WANDB_API_KEY \
   -e WANDB_PROJECT=3d_diffuser_actor_debug \
   -e DIFFUSER_ACTOR_ROOT=/workspace \
   -e PERACT_DATA=/workspace/data \
   -e POINTATTN_ROOT=/pointattn \
   -v $DIFFUSER_ACTOR_ROOT:/workspace \
   -v $POINTATTN_ROOT:/pointattention \
   -v $PERACT_DATA/test/:/workspace/data/test/ \
   -v $PERACT_DATA/instructions.pkl:/workspace/data/instructions.pkl \
   --shm-size=32gb oddtoddler400/3d_diffuser_actor:0.0.3)

# ============================================================
# Run training command
# ============================================================
docker exec -t $id /bin/bash -c "source scripts/slurm/startup-hook.sh && cd /workspace/ && \
    python data_preprocessing/rearrange_rlbench_demos.py --root_dir /workspace/data/test/ && \
    CUDA_LAUNCH_BLOCKING=1 xvfb-run -a python online_evaluation_rlbench/evaluate.py \
    --tasks $task \
    --checkpoint $checkpoint \
    --headless 1 \
    --hyper_params_file $hparams \
    --test_model $test_model \
    --cameras $cameras \
    --verbose 1 \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/pointattn_lang_enhanced_ipa/$exp_log_dir/$run_log_dir/seed$seed/$task.json  \
    --instructions $instructions \
    --variations $variations \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04"

docker stop $id