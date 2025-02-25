import re
import sys
import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

naming_dict = {
    "embedding_dim": "C",
    "batch_size": "B",
    "lr": "lr",
    "num_history": "H",
    "diffusion_timesteps": "DT",
    "rot_noise": "RN",
    "pos_noise": "PN",
    "pcd_noise": "PCDN",
    "fps_subsampling_factor": "FPS",
    "use_center_distance": "UCD",
    "use_center_projection": "UCP",
    "use_vector_projection": "UVP",
    "add_center": "AC",
    "feature_res": "FR",
    "feature_type": "FT",
    "distance_scale": "DS",
    "use_adaln": "ADALN",
}

def load_arguments_class(file_path: str):
    """
    Dynamically load the Arguments class from the given file.
    
    Args:
        file_path: Path to the Python file containing the Arguments class
        
    Returns:
        The Arguments class object
    """
    # Extract directory and filename
    file_path = Path(file_path).resolve()
    module_name = file_path.stem
    
    # Add the directory to sys.path temporarily
    sys.path.insert(0, str(file_path.parent))
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for the Arguments class
        if hasattr(module, 'Arguments'):
            return getattr(module, 'Arguments')
        else:
            raise AttributeError(f"Could not find Arguments class in {file_path}")
    finally:
        # Remove the directory from sys.path
        sys.path.pop(0)

def get_required_args(arguments_class) -> set:
    """
    Identify which arguments are required (no default value).
    
    Args:
        arguments_class: The Arguments class
        
    Returns:
        Set of required argument names
    """
    required_args = set()
    
    # Inspect annotations and defaults
    for name, annotation in arguments_class.__annotations__.items():
        # If the attribute doesn't have a default value in the class
        if not hasattr(arguments_class, name):
            required_args.add(name)
            
    return required_args

def generate_run_log_dir(arg_defaults: Dict[str, Any]) -> str:
    """
    Generate the run_log_dir based on the argument defaults.
    
    Args:
        arg_defaults: Dictionary of argument names to their default values

    Returns:
        String containing the run_log_dir
    """

    name = arg_defaults["name"]
    name += f"_$task_desc"
    for arg, prefix in naming_dict.items():
        if arg in arg_defaults.keys():
            name += f"-{prefix}${arg}"
    return name

def get_argument_defaults(arguments_class, args) -> Dict[str, Any]:
    """
    Get the default values for all arguments.
    
    Args:
        arguments_class: The Arguments class
        
    Returns:
        Dictionary mapping argument names to their default values
    """
    arg_defaults = {}
    required_args = get_required_args(arguments_class)
    
    # Get all class attributes that are part of the annotations
    for name in arguments_class.__annotations__:
        if getattr(args, name, None) is not None:
            arg_defaults[name] = getattr(args, name)
        elif name in required_args:
            arg_defaults[name] = "DEFINE_BY_USER"
        elif name == "exp_log_dir":
            arg_defaults[name] = "$(./scripts/utils/get_log_path.sh)"            
        elif hasattr(arguments_class, name):
            arg_defaults[name] = getattr(arguments_class, name)

    arg_defaults["run_log_dir"] = generate_run_log_dir(arg_defaults)
    arg_defaults["variations"] = '$(echo {0..199})'
            
    return arg_defaults, required_args

def format_value_for_bash(value: Any) -> str:
    """
    Format a Python value for bash script variable assignment.
    
    Args:
        value: The Python value to format
        
    Returns:
        String representation suitable for bash
    """
    if value == "DEFINE_BY_USER":
        return value
    
    if isinstance(value, tuple):
        # Format tuple as space-separated string
        return " ".join(str(x) for x in value)
    elif isinstance(value, Path):
        # Convert Path to string
        return str(value)
    elif isinstance(value, str):
        # Keep string as is, bash will handle it
        return value
    elif value is None:
        return '""'  # Empty string for None
    else:
        # Convert other types (int, float, etc.) to string
        return str(value)
def categorize_arguments(arg_defaults: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    categories = {
        "RLBench": {"cameras", "image_size", "max_episodes_per_task", "instructions", "tasks", "variations", "accumulate_grad_batches", "gripper_loc_bounds", "gripper_loc_bounds_buffer"},
        "Logging": {"val_freq", "base_log_dir", "exp_log_dir", "run_log_dir", "name", },
        "Training Parameters": {"num_workers", "batch_size", "batch_size_val", "cache_size", "cache_size_val", "lr", "wd", "train_iters", "val_iters", "max_episode_length", "seed", "checkpoint", "resume", "eval_only"},
        "Dataset Augmentations": {"rot_noise", "pos_noise", "pcd_noise", "image_rescale", "dense_interpolation", "interpolation_length"},
        "Model Parameters": set(),
    }

    categorized_args = {category: {} for category in categories}
    
    for arg_name, arg_value in arg_defaults.items():
        if arg_value == "DEFINE_BY_USER":
            continue
        found = False
        for category, args in categories.items():
            if arg_name in args:
                categorized_args[category][arg_name] = arg_value
                found = True
                break
        if not found:
            categorized_args["Model Parameters"][arg_name] = arg_value
    
    return categorized_args

def generate_train_sh(arg_defaults: Dict[str, Any], required_args, main_py: str, example_values: Optional[Dict[str, str]] = None) -> str:
    """
    Generate a train.sh script with all arguments defined as variables.
    
    Args:
        arg_defaults: Dictionary of argument names to their default values
        example_values: Optional dictionary of example values for required args
        
    Returns:
        String containing the bash script
    """
    categorized_args = categorize_arguments(arg_defaults)
    script = ["#!/bin/bash", 
              ""]
    
    # Add comment about required arguments
    if len(required_args) > 0:
        script.append("# ============================================================")
        script.append("# REQUIRED: You must set values for these variables")
        script.append("# ============================================================")
        
        for arg_name, arg_value in arg_defaults.items():
            if arg_name not in required_args:
                continue
            if arg_value is not None:
                script.append(f"{arg_name}=\"{arg_value}\"  # REQUIRED")
            else:
                script.append(f"{arg_name}=\"\"  # REQUIRED - Set this value")
        
        script.append("")
        script.append("# ============================================================")
        script.append("# Optional: You can modify these default values")
        script.append("# ============================================================")
    
    # Define variables for arguments with default values
    for category, args in categorized_args.items():
        script.append(f"# {category}")
        for arg_name, arg_value in args.items():
            if arg_name in required_args:
                continue

            if arg_name == "run_log_dir":
                continue

            formatted_value = format_value_for_bash(arg_value)
            
            # Handle special cases
            if arg_name == "checkpoint":
                script.append(f'#{arg_name}="" # Set this value to resume training')
            elif isinstance(arg_value, tuple) or arg_name in ["tasks", "cameras"]:
                script.append(f'{arg_name}="{formatted_value}"')
            else:
                script.append(f'{arg_name}={formatted_value}')
        script.append("")

    script.extend([
        'task_list=($tasks)',
        'if [ ${#task_list[@]} -gt 1 ]; then',
        '    task_desc="multitask"',
        'else',
        '    task_desc=${task_list[0]}',
        'fi',
        '',
        f'run_log_dir={arg_defaults["run_log_dir"]}',
        ''])

    
    # Add additional configuration
    script.extend([
        "",
        "# ============================================================",
        "# Configuration settings",
        "# ============================================================",
        "ngpus=$(python3 utils/count_cuda_devices.py)",
        "CUDA_LAUNCH_BLOCKING=1",
        "",
        "# ============================================================",
        "# Run training command",
        "# ============================================================",
        "torchrun --nproc_per_node $ngpus --master_port $RANDOM \\",
        f"    {main_py} \\"
    ])
    
    # Add command line arguments using the variables
    for i, args in enumerate(required_args):
        script.append(f"    --{args} ${{{args}}} \\")

    for category, args in categorized_args.items():
        for i, (arg_name, arg_value) in enumerate(args.items()):
            if arg_name in required_args:
                continue
            if arg_name == "checkpoint":
                continue
            else:
                script.append(f"    --{arg_name} ${{{arg_name}}} \\")

    if  "checkpoint" in arg_defaults.keys():
        script.append(f"#    --checkpoint $checkpoint # Set this value to resume training \\")
    
    # Remove the trailing backslash from the last line
    script[-1] = script[-1][:-2]
    
    return "\n".join(script)

def generate_slurm_sh(arg_defaults: Dict[str, Any], required_args, main_py: str, example_values: Optional[Dict[str, str]] = None) -> str:
    """
    Generate a train.sh script with all arguments defined as variables.
    
    Args:
        arg_defaults: Dictionary of argument names to their default values
        example_values: Optional dictionary of example values for required args
        
    Returns:
        String containing the bash script
    """
    categorized_args = categorize_arguments(arg_defaults)
    script = ["#!/bin/bash",
        "#SBATCH -t 24:00:00",
        "#SBATCH -c 4",
        "#SBATCH --mem=32G",
        "#SBATCH -p gpu",
        "#SBATCH --array=0-4%1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --output=train_logs/slurm_logs/%A_train/%a.out",
        f"#SBATCH -J {arg_defaults['name']}",]
    
    # Add comment about required arguments
    if len(required_args) > 0:
        script.append("# ============================================================")
        script.append("# REQUIRED: You must set values for these variables")
        script.append("# ============================================================")
        
        for arg_name, arg_value in arg_defaults.items():
            if arg_name not in required_args:
                continue
            if arg_value is not None:
                script.append(f"{arg_name}=\"{arg_value}\"  # REQUIRED")
            else:
                script.append(f"{arg_name}=\"\"  # REQUIRED - Set this value")
        
        script.append("")
        script.append("# ============================================================")
        script.append("# Optional: You can modify these default values")
        script.append("# ============================================================")
    
    # Define variables for arguments with default values
    for category, args in categorized_args.items():
        script.append(f"# {category}")
        for arg_name, arg_value in args.items():
            if arg_name in required_args:
                continue

            if arg_name == "run_log_dir":
                continue

            formatted_value = format_value_for_bash(arg_value)
            
            # Handle special cases
            if arg_name == "checkpoint":
                script.append(f'#{arg_name}="" # Set this value to resume training')
            elif isinstance(arg_value, tuple) or arg_name in ["tasks", "cameras"]:
                script.append(f'{arg_name}="{formatted_value}"')
            else:
                script.append(f'{arg_name}={formatted_value}')
        script.append("")

    script.extend([
        'task_list=($tasks)',
        'if [ ${#task_list[@]} -gt 1 ]; then',
        '    task_desc="multitask"',
        'else',
        '    task_desc=${task_list[0]}',
        'fi',
        '',
        f'run_log_dir={arg_defaults["run_log_dir"]}',
        ''])

    
    # Add additional configuration
    script.extend([
        "",
        "# ============================================================",
        "# Configuration settings",
        "# ============================================================",
        "ngpus=$(python3 utils/count_cuda_devices.py)",
        "CUDA_LAUNCH_BLOCKING=1",
        ""]
    )
    script.extend(["# ============================================================",
        "# Set up log directory",
        "# ============================================================",
        'LOG_DIR_FILE=~/3d_diffuser_actor/train_logs/slurm_logs/${SLURM_ARRAY_JOB_ID}_train/log_dir.txt',
        'if [ -n "$log_dir" ] && [ ! -f $LOG_DIR_FILE ]; then',
        '    echo "$log_dir" > $LOG_DIR_FILE',
        'fi',
        'if [ $SLURM_ARRAY_TASK_ID -gt 0 ] || [ -n "$log_dir" ]; then',
        '    log_dir=$(cat $LOG_DIR_FILE)',
        '    kwargs="$kwargs --resume 1"',
        '    if [ -f "$log_dir/last.pth" ]; then',
        '        kwargs="$kwargs --checkpoint $log_dir/last.pth"',
        '    fi',
        'else',
        '    echo "$base_log_dir/$main_dir/$run_log_dir" > $LOG_DIR_FILE',
        'fi'])

    script.extend(['echo "Starting docker container"',
    'id=$(docker run -dt \\',
    '    -e WANDB_API_KEY=$WANDB_API_KEY \\',
    '    -e WANDB_PROJECT=3d_diffuser_actor_debug \\',
    '    -v ~/3d_diffuser_actor:/workspace \\',
    '    -v ~/pointattention/:/pointattention \\',
    '    -v /home/share/3D_attn_felix/Peract_packaged/:/workspace/data/Peract_packaged/ \\',
    '    -v /home/share/3D_attn_felix/peract/instructions.pkl:/workspace/data/peract/instructions.pkl \\',
    '    --shm-size=32gb oddtoddler400/3d_diffuser_actor:0.0.3)'])

    
    script.extend([
        "# ============================================================",
        "# Run training command",
        "# ============================================================",
    ])
    
    script.extend(['docker exec -t $id /bin/bash -c "source scripts/slurm/startup-hook.sh && cd /workspace/ &&',
        '    CUDA_LAUNCH_BLOCKING=1 torchrun \\',
        '    --nproc_per_node $ngpus \\',
        '    --master_port $RANDOM \\',
        f'    {main_py} \\'
    ])

    # Add command line arguments using the variables
    for i, args in enumerate(required_args):
        script.append(f"    --{args} ${{{args}}} \\")

    for category, args in categorized_args.items():
        for i, (arg_name, arg_value) in enumerate(args.items()):
            if arg_name in required_args:
                continue
            if arg_name == "checkpoint":
                continue
            else:
                script.append(f"    --{arg_name} ${{{arg_name}}} \\")

    if  "checkpoint" in arg_defaults.keys():
        script.append(f"#    --checkpoint $checkpoint # Set this value to resume training \\")
    
    # Remove the trailing backslash from the last line
    script[-1] = script[-1][:-2] + '"'

    script.extend(["docker stop $id"])
    
    return "\n".join(script)

def extract_example_values_from_template(template_path: str) -> Dict[str, str]:
    """
    Extract example values from a template file.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        Dictionary of argument names to example values
    """
    examples = {}
    
    if not template_path or not Path(template_path).exists():
        return examples
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Look for variable assignments in the template
    var_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)=(.+)$', re.MULTILINE)
    for match in var_pattern.finditer(content):
        name, value = match.groups()
        # Clean up the value (remove quotes, etc.)
        value = value.strip().strip('"\'')
        examples[name] = value
    
    return examples

def main():
    parser = argparse.ArgumentParser(description="Generate train.sh from main.py arguments")
    parser.add_argument("main_py", help="Path to main.py file")
    parser.add_argument("--template", help="Path to template train.sh file for example values")
    parser.add_argument("--output_dir", default="scripts", help="Output file path")
    parser.add_argument("--output", default="train.sh", help="Output file path")
    parser.add_argument("--gripper_loc_bounds", default=None, help="Gripper location bounds")
    parser.add_argument("--instructions", default=None, help="Instructions")
    parser.add_argument("--dataset", default=None, help="Dataset")
    parser.add_argument("--valset", default=None, help="Validation set")
    parser.add_argument("--tasks", default=None, nargs="+", help="Tasks")
        
    args = parser.parse_args()

    if args.tasks is not None:
        args.tasks = " ".join(args.tasks)
    
    # try:
    # Load the Arguments class
    arguments_class = load_arguments_class(args.main_py)
    
    # Get default values for all arguments
    arg_defaults, required_args = get_argument_defaults(arguments_class, args)
    
    # Get example values from template if provided
    example_values = extract_example_values_from_template(args.template) if args.template else None
    
    # Generate the train.sh script
    print(f"Generating {args.output}")

    train_sh = generate_train_sh(arg_defaults, required_args, args.main_py, example_values)
    slurm_sh = generate_slurm_sh(arg_defaults, required_args, args.main_py, example_values)
    
    train_output_path = Path(args.output_dir) / "train" / args.output
    slurm_output_path = Path(args.output_dir) / "slurm" / args.output
    with open(train_output_path, 'w') as f:
        f.write(train_sh)
    with open(slurm_output_path, 'w') as f:
        f.write(slurm_sh)
    
    print(f"Successfully generated {args.output}")
    
    # Report required arguments
    required_args = [name for name, value in arg_defaults.items() if value == "DEFINE_BY_USER"]
    if required_args:
        print(f"\nNOTE: The following arguments need to be defined in the script: {', '.join(required_args)}")
        
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return 1
    
    return 0

if __name__ == "__main__":
    exit(main())