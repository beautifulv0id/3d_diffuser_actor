#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --output=eval_logs/slurm_logs/%j_train.out
#SBATCH -J eval_pointattn

docker_run_script=$1
kwargs=${@:2}

cd ~/3d_diffuser_actor

echo "Starting docker container"
id=$(docker run -dt  -v ~/3d_diffuser_actor:/workspace \
    -v ~/pointattention/:/pointattention \
    -v /home/share/3D_attn_felix/train_logs:/workspace/train_logs \
    -v /home/share/3D_attn_felix/peract/test/:/workspace/data/peract/test/ \
    -v /home/share/3D_attn_felix/peract/instructions.pkl:/workspace/data/peract/instructions.pkl \
    --shm-size=32gb oddtoddler400/3d_diffuser_actor:0.0.3)

echo "Container ID: $id"
echo "Running $docker_run_script in container"
docker exec -t $id /bin/bash -c "source slurm_scripts/startup-hook.sh &&
                        bash $docker_run_script $kwargs"
docker stop $id



