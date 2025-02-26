#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -c 3
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=eval_logs/slurm_logs/%j.out
#SBATCH -J preprocess_rlbench_instructions

id=$(docker run -dt \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_PROJECT=3d_diffuser_actor_debug \
    -v $DIFFUSER_ACTOR_ROOT:/workspace \
    -v $POINTATTN_ROOT:/pointattention \
    -v $PERACT_DATA/Peract_packaged/:/workspace/data/Peract_packaged/ \
    -v $PERACT_DATA/instructions.pkl:/workspace/data/peract/instructions.pkl \
    --shm-size=32gb oddtoddler400/3d_diffuser_actor:0.0.3)

echo "Container ID: $id"
docker exec -t $id /bin/bash -c "source slurm_scripts/startup-hook.sh && cd /workspace/ &&
                        xvfb-run -a python data_preprocessing/preprocess_rlbench_instructions.py --output /workspace/data/peract/instructions.pkl"
docker stop $id

