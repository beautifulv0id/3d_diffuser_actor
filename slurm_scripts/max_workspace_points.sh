#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -c 3
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH --output=eval_logs/slurm_logs/%j.out
#SBATCH -J max_workspace_points

tasks=$1

id=$(docker run -dt \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_PROJECT=3d_diffuser_actor_debug \
    -v ~/3d_diffuser_actor:/workspace \
    -v ~/pointattention/:/pointattention \
    -v /home/share/3D_attn_felix/Peract_packaged/:/workspace/data/Peract_packaged/ \
    -v /home/share/3D_attn_felix/peract/instructions.pkl:/workspace/data/peract/instructions.pkl \
    --shm-size=16gb oddtoddler400/3d_diffuser_actor:0.0.3)

echo "Container ID: $id"
docker exec -t $id /bin/bash -c "source slurm_scripts/startup-hook.sh && cd /workspace/ &&
                        python utils/max_workspace_points.py \
                        --output /workspace/tasks/max_workspace_points.json \
                        --dataset /workspace/data/Peract_packaged/train \
                        --valset /workspace/data/Peract_packaged/val \
                        --tasks $tasks"
docker stop $id

