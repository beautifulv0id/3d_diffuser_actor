exp=point_attn

tasks=(
    close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
)
data_dir=/data/peract/test/
num_episodes=100
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
max_tries=2
verbose=1
single_task_gripper_loc_bounds=0
cameras="left_shoulder,right_shoulder,wrist,front"
seed=0
log_dir="train_logs/2025.01.22/20.45.18_Actor_18Peract_100Demo_multitask/diffusion_multitask-C120-B4-lr1e-4-DI1-2-H3-DT100"
checkpoint=$log_dir/best.pth
hyper_params_file=$log_dir/hparams.json
action_dim=8
test_model="pointattn_self_attn"
num_ckpts=${#tasks[@]}

# Maximum number of parallel processes
max_parallel=1
current_jobs=0

for ((i=0; i<$num_ckpts; i++)); do
    echo "Evaluating ${tasks[$i]}"
    CUDA_LAUNCH_BLOCKING=1 xvfb-run -a python online_evaluation_rlbench/evaluate_policy_from_hparams.py \
    --tasks ${tasks[$i]} \
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
    --instructions /data/peract/instructions.pkl \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04 &

    ((current_jobs++))

    # Wait for some jobs to finish if the max limit is reached
    if ((current_jobs >= max_parallel)); then
        wait -n  # Wait for any background job to finish
        ((current_jobs--))
    fi

done

