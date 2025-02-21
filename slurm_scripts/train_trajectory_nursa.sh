#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --array=0-4%1
#SBATCH --gres=gpu:1
#SBATCH --output=train_logs/slurm_logs/%A_train/%a.out
#SBATCH -J train_3dda_ursa

echo "Command Line Arguments: $@"

# Load modules
config_name=NURSA
main_dir=$(./scripts/get_log_path.sh $config_name)

# Default parameter values
dataset="data/Peract_packaged/train/"
valset="data/Peract_packaged/val/"
instructions="data/peract/instructions.pkl"
gripper_loc_bounds="tasks/18_peract_tasks_location_bounds.json"
variations=$(seq 0 199)

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

rot_noise=0.0
pos_noise=0.0
pcd_noise=0.0

# Model parameters
diffusion_timesteps=100
num_history=3
fps_subsampling_factor=5
rotation_parametrization='6D'
use_instruction=1
relative=0

# Logging parameters
base_log_dir="train_logs"
if [ ${#tasks[@]} -gt 1 ]; then
    task_desc="multitask"
else
    task_desc=${tasks[0]}
fi

run_log_dir="diffusion_nursa_$task_desc-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-RN$rot_noise-PN$pos_noise-PCDN$pcd_noise-fps$fps_subsampling_factor"

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
    --fps_subsampling_factor) fps_subsampling_factor="$2"; shift; shift;;
    --log_dir) log_dir="$2"; shift; shift;;
    *) echo "Unknown option: $key"; exit 1;;
  esac
done

# Set up kwargs
kwargs="--dataset $dataset \
        --tasks ${tasks[@]} \
        --valset $valset \
        --instructions $instructions \
        --use_instruction $use_instruction \
        --rotation_parametrization $rotation_parametrization \
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
        --variations ${variations[@]} \
        --lr $lr \
        --num_history $num_history \
        --cameras left_shoulder right_shoulder wrist front \
        --max_episodes_per_task -1 \
        --quaternion_format $quaternion_format \
        --fps_subsampling_factor $fps_subsampling_factor \
        --rot_noise $rot_noise \
        --pos_noise $pos_noise \
        --pcd_noise $pcd_noise \
        --run_log_dir $run_log_dir"

# Count number of GPUs
ngpus=$(python3 scripts/helper/count_cuda_devices.py)

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
                        main_trajectory_nursa.py $kwargs"
docker stop $id
