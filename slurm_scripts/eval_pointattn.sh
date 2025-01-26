exp=point_attn

tasks=$1
data_dir=$PERACT_TEST_DIR
num_episodes=100
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
max_tries=2
verbose=1
single_task_gripper_loc_bounds=0
cameras="left_shoulder,right_shoulder,wrist,front"
seed=0
log_dir="train_logs/2025.01.23/19.12.58_PointAttn_18Peract_100Demo_multitask/pointattn_multitask-C120-B32-lr1e-4-DI1-2-H3-DT100"
checkpoint=$log_dir/best.pth
hyper_params_file=$log_dir/hparams.json
action_dim=8
test_model="pointattn"
echo $tasks
CUDA_LAUNCH_BLOCKING=1 xvfb-run -a python online_evaluation_rlbench/evaluate_policy_from_hparams.py \
    --tasks $tasks \
    --checkpoint $checkpoint \
    --headless 1 \
    --hyper_params_file $hyper_params_file \
    --test_model $test_model \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --instructions $PERACT_INSTRUCTIONS \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04 \
    --test_model pointattn 