config_name=PointAttnSelfAttn_18Peract_100Demo_multitask
main_dir=$(./scripts/get_log_path.sh $config_name)

dataset="/home/share/3D_attn_felix/Peract_packaged/train/"
valset="/home/share/3D_attn_felix/Peract_packaged/val/"
instructions="/home/share/3D_attn_felix/rlbench_instructions/instructions.pkl"
max_workspace_points="tasks/max_workspace_points.json"

ngpus=$(python3 scripts/helper/count_cuda_devices.py)

rot_noise=0.0
pos_noise=0.0
pcd_noise=0.0

dense_interpolation=1
interpolation_length=2
tasks=(open_drawer)

lr=1e-4
B=1
C=120
quaternion_format=xyzw
batch_size_val=12
cache_size=0
cache_size_val=0
val_freq=1000
train_iters=200000

# Model parameters
diffusion_timesteps=10
num_history=3
use_normals=0
rot_factor=1
gripper_depth=2
decoder_depth=4
decoder_dropout=0.0
distance_scale=1.0
use_adaln=1
feature_res="res3"
fps_subsampling_factor=5
gripper_history_as_points=0
use_center_distance=1
use_center_projection=1
use_vector_projection=0
add_center=1
feature_type="sinusoid"
crop_workspace=1
point_embedding_dim=120


if [ ${#tasks[@]} -gt 1 ]; then
    task_desc="multitask"
else
    task_desc=${tasks[0]}
fi

run_log_dir="pointattn_esa_$task_desc-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-RN$rot_noise-PN$pos_noise-PCDN$pcd_noise-drop$decoder_dropout-DS$distance_scale-ADALN$use_adaln-$feature_res-fps$fps_subsampling_factor-HP$gripper_history_as_points-CD$use_center_distance-CP$use_center_projection-VP$use_vector_projection-AC$add_center-FT$feature_type"

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_pointattn_efficient_self_attn.py \
    --tasks ${tasks[@]} \
    --dataset $dataset \
    --valset $valset \
    --instructions $instructions \
    --max_workspace_points $max_workspace_points \
    --crop_workspace $crop_workspace \
    --point_embedding_dim $point_embedding_dim \
    --rot_noise $rot_noise \
    --pos_noise $pos_noise \
    --pcd_noise $pcd_noise \
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
    --run_log_dir $run_log_dir \
    --fps_subsampling_factor $fps_subsampling_factor \
    --gripper_history_as_points $gripper_history_as_points \
    --use_center_distance $use_center_distance \
    --use_center_projection $use_center_projection \
    --use_vector_projection $use_vector_projection \
    --add_center $add_center \
    --feature_res $feature_res \
    --feature_type $feature_type \
    --distance_scale $distance_scale \
    --use_adaln $use_adaln
```