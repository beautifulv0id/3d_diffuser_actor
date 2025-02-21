exp=point_attn

# tasks=(
#     close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
# )
tasks=(
    stack_blocks
)
#  tasks=(
#    stack_cups
#)
#data_dir=/media/funk/INTENSO/300_RL_BENCH/data/peract/train/
#instruction_path=/media/funk/INTENSO/300_RL_BENCH/data/peract/instructions.pkl
num_eval_iters=1
data_dir=/media/funk/INTENSO/300_RL_BENCH/data/peract/test
instruction_path=/media/funk/INTENSO/300_RL_BENCH/peract_from_cluster/instructions.pkl
num_episodes=100
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
max_tries=10
verbose=1
single_task_gripper_loc_bounds=0
cameras="left_shoulder,right_shoulder,wrist,front"
seed=$1
checkpoint_raw_folder=/media/funk/INTENSO/300_RL_BENCH/0_evals/final_trains_29_01_evening/stack_blocks/ursa/09.48.55_PointAttn_18Peract_100Demo_multitask/diffusion_multitask-C120-B6-lr1e-4-DI1-2-H3-DT100
checkpoint=$checkpoint_raw_folder/best.pth
hyper_params_file=$checkpoint_raw_folder/hparams.json
action_dim=8

num_ckpts=${#tasks[@]}
for ((j=0; j<num_eval_iters; j++)); do
for ((i=0; i<$num_ckpts; i++)); do
    echo "Evaluating ${tasks[$i]}"
    CUDA_LAUNCH_BLOCKING=1 python online_evaluation_rlbench/evaluate_policy_from_hparams.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --headless 0 \
    --hyper_params_file $hyper_params_file \
    --test_model pointattn_25 \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file $checkpoint_raw_folder/${tasks[$i]}_eval1$1_$j.json  \
    --instructions $instruction_path \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04
done
done
