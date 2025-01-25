config_name=Actor_18Peract_100Demo_multitask
main_dir=$(./scripts/get_log_path.sh $config_name)

dataset=/data/Peract_packaged/train
valset=/data/Peract_packaged/val

lr=1e-4
dense_interpolation=1
interpolation_length=2
num_history=3
diffusion_timesteps=100
B=4
C=120
ngpus=$(python3 scripts/helper/count_cuda_devices.py)
quaternion_format=xyzw
use_normals=0
rot_factor=1
gripper_depth=2
decoder_depth=8
decoder_dropout=0.2

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_pointattn_self_attn.py \
    --tasks place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap \
    --dataset $dataset \
    --valset $valset \
    --instructions /data/rlbench_instructions/instructions.pkl \
    --gripper_loc_bounds tasks/18_peract_tasks_location_bounds.json \
    --num_workers 1 \
    --train_iters 600000 \
    --embedding_dim $C \
    --diffusion_timesteps $diffusion_timesteps \
    --val_freq 500 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 10000 \
    --cache_size_val 0 \
    --keypose_only 1 \
    --variations {0..199} \
    --lr $lr\
    --num_history $num_history \
    --cameras left_shoulder right_shoulder wrist front\
    --max_episodes_per_task -1 \
    --quaternion_format $quaternion_format \
    --use_normals $use_normals \
    --rot_factor $rot_factor \
    --gripper_depth $gripper_depth\
    --decoder_depth $decoder_depth\
    --decoder_dropout $decoder_dropout\
    --run_log_dir pointattn_sa_multitask-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps \
    --fps_subsampling_factor 10