import pickle
from typing import Dict, Optional, Sequence
from pathlib import Path
import json
import torch
import numpy as np
import torch.nn as nn

Instructions = Dict[str, Dict[int, torch.Tensor]]

def apply_to_module(model: nn.Module, target_module: type, func):
    """
    Recursively applies `func` to all instances of `target_module` in `model`.
    
    Args:
        model (nn.Module): The PyTorch model to traverse.
        target_module (type): The type of module to apply `func` to (e.g., nn.Conv2d).
        func (callable): A function that takes a module as input and performs an operation.
    """
    for name, module in model.named_children():
        if isinstance(module, target_module):
            func(module)
        else:
            apply_to_module(module, target_module, func)  # Recurse into submodules


def round_floats(o):
    if isinstance(o, float): return round(o, 2)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


def normalise_quat(x: torch.Tensor):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


def get_gripper_loc_bounds(path: str, buffer: float = 0.0, task: Optional[str] = None):
    gripper_loc_bounds = json.load(open(path, "r"))
    if task is not None and task in gripper_loc_bounds:
        gripper_loc_bounds = gripper_loc_bounds[task]
        gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
        gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    else:
        # Gripper workspace is the union of workspaces for all tasks
        gripper_loc_bounds = json.load(open(path, "r"))
        gripper_loc_bounds_min = np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]), axis=0) - buffer
        gripper_loc_bounds_max = np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]), axis=0) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
    return gripper_loc_bounds

def load_max_workspace_points(path: str, task = None):
    max_workspace_points = json.load(open(path, "r"))
    if task is None:
        max_workspace_points = np.max(np.stack([np.array(points) for points in max_workspace_points.values()]), axis=0)
    else:
        max_workspace_points = np.array(max_workspace_points[task])
    return max_workspace_points

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def norm_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.linalg.norm(tensor, ord=2, dim=-1, keepdim=True)


def load_instructions(
    instructions: Optional[Path],
    tasks: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[int]] = None,
) -> Optional[Instructions]:
    if instructions is not None:
        with open(instructions, "rb") as fid:
            data: Instructions = pickle.load(fid)
        if tasks is not None:
            data = {task: var_instr for task, var_instr in data.items() if task in tasks}
        if variations is not None:
            data = {
                task: {
                    var: instr for var, instr in var_instr.items() if var in variations
                }
                for task, var_instr in data.items()
            }
        return data
    return None