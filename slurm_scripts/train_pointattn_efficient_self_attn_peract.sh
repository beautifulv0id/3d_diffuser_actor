#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --array=0-4%1
#SBATCH --gres=gpu:1
#SBATCH --output=train_logs/slurm_logs/%A_train/%a.out
#SBATCH -J train_pointattn_efficient_self_attn_peract

echo "Command Line Arguments: $@"

# Load modules
config_name=PointAttn
main_dir=$(./scripts/get_log_path.sh $config_name)

# Default parameter values
dataset="data/Peract_packaged/train/"
valset="data/Peract_packaged/val/"
instructions="data/peract/instructions.pkl"
gripper_loc_bounds="tasks/18_peract_tasks_location_bounds.json"

rot_noise=0.0
pos_noise=0.0
pcd_noise=0.0

dense_interpolation=1
interpolation_length=2
tasks=(stack_blocks)

lr=1e-4
B=8
C=120
quaternion_format=xyzw
batch_size_val=12
cache_size=0
cache_size_val=0
val_freq=1000
train_iters=200000

# Model parameters
diffusion_timesteps=100
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
gripper_history_as_points=1
use_center_distance=1
use_center_projection=1
use_vector_projection=0
add_center=1
feature_type="sinusoid"

base_log_dir="train_logs"

if [ ${#tasks[@]} -gt 1 ]; then
    task_desc="multitask"
else
    task_desc=${tasks[0]}
fi

run_log_dir="pointattn_esa_$task_desc-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-RN$rot_noise-PN$pos_noise-PCDN$pcd_noise-drop$decoder_dropout-DS$distance_scale-ADALN$use_adaln-$feature_res-fps$fps_subsampling_factor-HP$gripper_history_as_points-CD$use_center_distance-CP$use_center_projection-VP$use_vector_projection-AC$add_center-FT$feature_type"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dataset) dataset="$2"; shift; shift;;
    --valset) valset="$2"; shift; shift;;
    --instructions) instructions="$2"; shift; shift;;
    --gripper_loc_bounds) gripper_loc_bounds="$2"; shift; shift;;
    --rot_noise) rot_noise="$2"; shift; shift;;
    --pos_noise) pos_noise="$2"; shift; shift;;
    --pcd_noise) pcd_noise="$2"; shift; shift;;
    --dense_interpolation) dense_interpolation="$2"; shift; shift;;
    --interpolation_length) interpolation_length="$2"; shift; shift;;
    --tasks) tasks=($2); shift; shift;;
    --lr) lr="$2"; shift; shift;;
    --batch_size) B="$2"; shift; shift;;
    --embedding_dim) C="$2"; shift; shift;;
    --batch_size_val) batch_size_val="$2"; shift; shift;;
    --cache_size) cache_size="$2"; shift; shift;;
    --cache_size_val) cache_size_val="$2"; shift; shift;;
    --val_freq) val_freq="$2"; shift; shift;;
    --train_iters) train_iters="$2"; shift; shift;;
    --diffusion_timesteps) diffusion_timesteps="$2"; shift; shift;;
    --num_history) num_history="$2"; shift; shift;;
    --decoder_dropout) decoder_dropout="$2"; shift; shift;;
    --distance_scale) distance_scale="$2"; shift; shift;;
    --use_adaln) use_adaln="$2"; shift; shift;;
    --feature_res) feature_res="$2"; shift; shift;;
    --fps_subsampling_factor) fps_subsampling_factor="$2"; shift; shift;;
    --feature_type) feature_type="$2"; shift; shift;;
    --gripper_depth) gripper_depth="$2"; shift; shift;;
    --decoder_depth) decoder_depth="$2"; shift; shift;;
    --gripper_history_as_points) gripper_history_as_points="$2"; shift; shift;;
    --use_center_distance) use_center_distance="$2"; shift; shift;;
    --use_center_projection) use_center_projection="$2"; shift; shift;;
    --use_vector_projection) use_vector_projection="$2"; shift; shift;;
    --add_center) add_center="$2"; shift; shift;;
    --log_dir) log_dir="$2"; shift; shift;;
    *) echo "Unknown option: $key"; exit 1;;
  esac
done

# Set up kwargs
kwargs="--dataset $dataset \
        --tasks ${tasks[@]} \
        --valset $valset \
        --instructions $instructions \
        --gripper_loc_bounds $gripper_loc_bounds \
        --num_workers 1 \
        --train_iters $train_iters \
        --embedding_dim $C \
        --diffusion_timesteps $diffusion_timesteps \
        --val_freq $val_freq \
        --dense_interpolation $dense_interpolation \
        --interpolation_length $interpolation_length \
        --exp_log_dir $main_dir \
        --base_log_dir $base_log_dir \
        --batch_size $B \
        --batch_size_val $batch_size_val \
        --cache_size $cache_size \
        --cache_size_val $cache_size_val \
        --keypose_only 1 \
        --variations {0..199} \
        --lr $lr \
        --num_history $num_history \
        --cameras left_shoulder right_shoulder wrist front \
        --max_episodes_per_task -1 \
        --quaternion_format $quaternion_format \
        --use_normals $use_normals \
        --rot_factor $rot_factor \
        --gripper_depth $gripper_depth \
        --decoder_depth $decoder_depth \
        --decoder_dropout $decoder_dropout \
        --distance_scale $distance_scale \
        --use_adaln $use_adaln \
        --feature_res $feature_res \
        --fps_subsampling_factor $fps_subsampling_factor \
        --rot_noise $rot_noise \
        --pos_noise $pos_noise \
        --pcd_noise $pcd_noise \
        --gripper_history_as_points $gripper_history_as_points \
        --use_center_distance $use_center_distance \
        --use_center_projection $use_center_projection \
        --use_vector_projection $use_vector_projection \
        --run_log_dir $run_log_dir \
        --add_center $add_center \
        --feature_type $feature_type"

# Save log directory to file
LOG_DIR_FILE=~/3d_diffuser_actor/train_logs/slurm_logs/${SLURM_ARRAY_JOB_ID}_train/log_dir.txt
if [ -n "$log_dir" ] && [ ! -f $LOG_DIR_FILE ]; then
    echo "$log_dir" > $LOG_DIR_FILE
fi
if [ $SLURM_ARRAY_TASK_ID -gt 0 ] || [ -n "$log_dir" ]; then
    log_dir=$(cat $LOG_DIR_FILE)
    kwargs="$kwargs --resume 1"
    if [ -f "$log_dir/last.pth" ]; then
        kwargs="$kwargs --checkpoint $log_dir/last.pth"
    fi
else
    echo "$base_log_dir/$main_dir/$run_log_dir" > $LOG_DIR_FILE
fi

# Count number of GPUs
ngpus=$(python3 scripts/helper/count_cuda_devices.py)
# Start docker container
cd ~/3d_diffuser_actor
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Args: "
echo $kwargs

echo "Starting docker container"
id=$(docker run -dt \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_PROJECT=3d_diffuser_actor_debug \
    -v ~/3d_diffuser_actor:/workspace \
    -v ~/pointattention/:/pointattention \
    -v /home/share/3D_attn_felix/Peract_packaged/:/workspace/data/Peract_packaged/ \
    -v /home/share/3D_attn_felix/peract/instructions.pkl:/workspace/data/peract/instructions.pkl \
    --shm-size=32gb oddtoddler400/3d_diffuser_actor:0.0.3)

echo "Container ID: $id"
docker exec -t $id /bin/bash -c "source slurm_scripts/startup-hook.sh && cd /workspace/ &&
                        CUDA_LAUNCH_BLOCKING=1 torchrun \
                        --nproc_per_node $ngpus \
                        --master_port $RANDOM \
                        main_pointattn_efficient_self_attn.py $kwargs"
docker stop $id
