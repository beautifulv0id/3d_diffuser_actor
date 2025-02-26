import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import os

def generate_det(script_name) -> str:
    script = [
        f'name: {script_name}',
        'workspace: PI_Chalvatzaki',
        'project: pointattention',
        '',
        'max_restarts: 0',
        '',
        'bind_mounts:',
        '  - container_path: /pfss/mlde/mlde_ext',
        '    host_path: /pfss/mlde/mlde_ext',
        '    propagation: rprivate',
        '    read_only: false',
        '  - container_path: /pfss/mlde/workspaces/mlde_wsp_PI_Chalvatzaki',
        '    host_path: /pfss/mlde/workspaces/mlde_wsp_PI_Chalvatzaki',
        '  - container_path: /pfss/mlde/users/fh36jega',
        '    host_path: /pfss/mlde/users/fh36jega',
        '  - container_path: /shared',
        '    host_path: /pfss/mlde/workspaces/mlde_wsp_PI_Chalvatzaki/shared',
        '  - container_path: /run/determined/workdir',
        '    host_path: /pfss/mlde/workspaces/mlde_wsp_PI_Chalvatzaki/home/fh36jega',
        '  - container_path: /workspace/',
        '    host_path: /pfss/mlde/workspaces/mlde_wsp_PI_Chalvatzaki/home/fh36jega/3d_diffuser_actor',
        '  - container_path: /workspace/data/Peract_packaged',
        '    host_path: /pfss/mlde/workspaces/mlde_wsp_PI_Chalvatzaki/home/fh36jega/data/Peract_packaged',
        '  - container_path: /workspace/data/peract/instructions.pkl',
        '    host_path: /pfss/mlde/workspaces/mlde_wsp_PI_Chalvatzaki/home/fh36jega/data/peract/instructions.pkl',
        '  - container_path: /pointattention',
        '    host_path: /pfss/mlde/workspaces/mlde_wsp_PI_Chalvatzaki/home/fh36jega/pointattention',
        '',
        'environment:',
        '  environment_variables:',
        '    - WANDB_API_KEY=8009cee998358d908f42c2fce77f1ee094836701',
        '    - WANDB_PROJECT=3d_diffuser_actor_debug',
        '    - PERACT_DATA=/workspace/data/',
        '    - POINTATTN_ROOT=/pointattention',
        '  add_capabilities:',
        '    - IPC_LOCK',
        '  drop_capabilities: null',
        '  force_pull_image: false',
        '  image:',
        '    cpu: oddtoddler400/3d_diffuser_actor:0.0.3',
        '    cuda: oddtoddler400/3d_diffuser_actor:0.0.3',
        '',
        'resources:',
        '  devices:',
        '    - container_path: /dev/infiniband/',
        '      host_path: /dev/infiniband/',
        '      mode: mrw',
        '  resource_pool: 42_Compute',
        '  slots_per_trial: 1 # # GPUs',
        '',
        f'entrypoint: exec bash -c "cd /workspace && source scripts/slurm/startup-hook.sh && bash scripts/train/{script_name}.sh"',
        '',
        'searcher:',
        '    name: single',
        '    metric: validation_loss',
        '    max_length:',
        '      epochs: 1']
    return "\n".join(script)

def main():
    train_scripts = []
    for file in os.listdir("scripts/train"):
        if file.endswith(".sh"):
            train_scripts.append(file[:-3])    

    os.makedirs("det_scripts", exist_ok=True)

    for train_script in train_scripts:
        det_sh = generate_det(train_script)
        
        with open(f'det_scripts/{train_script}.yaml', 'w') as f:
            f.write(det_sh)
    
        print(f"Successfully generated {train_script}")
        
    return 0

if __name__ == "__main__":
    exit(main())