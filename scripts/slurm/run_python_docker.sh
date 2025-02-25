python_file=$1
kwargs=${@:2}

cd ~/3d_diffuser_actor
echo "Starting docker container"
id=$(docker run -dt  \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v ~/3d_diffuser_actor:/workspace \
    -v ~/pointattention/:/pointattention \
    -v /home/share/3D_attn_felix/train_logs:/workspace/train_logs \
    -v /home/share/3D_attn_felix/peract/test/:/data/peract/test/ \
    -v /home/share/3D_attn_felix/peract/instructions.pkl:/data/peract/instructions.pkl \
    --shm-size=8gb oddtoddler400/3d_diffuser_actor:0.0.3)
echo "Container ID: $id"
echo "Running $python_file in container"
docker exec -t $id /bin/bash -c "source slurm_scripts/startup-hook.sh &&
                        python $python_file $kwargs"
docker stop $id



