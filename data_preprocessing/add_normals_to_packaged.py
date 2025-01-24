import random
import itertools
from typing import Tuple, Dict, List
import pickle
from pathlib import Path
import json

import blosc
from tqdm import tqdm
import tap
import torch
import numpy as np
import einops
from rlbench.demo import Demo

from datasets.utils import (
    loader
)
from utils.common_utils import (
    load_instructions, get_gripper_loc_bounds
)

from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path
from time import time
import pickle
import blosc
import numpy as np
import open3d as o3d
import torch

def compute_normals(pcd, camera_view):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    normals = -(pcd - camera_view[None, :])
    init_normals = normals / (np.linalg.norm(normals, axis=-1)[..., None] + 1e-10)
    pcd_o3d.normals = o3d.utility.Vector3dVector(init_normals)

    pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    ## Align to Camera View
    normals = np.asarray(pcd_o3d.normals)
    # Compute the dot product between normals and view vectors
    dot_products = np.einsum('ij,ij->i', normals, init_normals)
    normals[dot_products < 0] *= -1

    return normals


def print_dict(x, indent=0, print_values=False):
    for k in x.keys():
        if isinstance(x[k], dict):
            print(" "*3*indent+k+":")
            print_dict(x[k], indent+1, print_values=print_values)
        else:
            if isinstance(x[k], torch.Tensor):
                print(" "*3*indent+str(k)+":", x[k].detach().cpu().numpy().shape, x[k].dtype)
                if print_values:
                    print( x[k][0].detach().cpu().numpy())
            else:
                print(" "*3*indent+str(k)+":", x[k])

def print_list(x, indent=0, print_values=False):
    for k in range(len(x)):
        if isinstance(x[k], list):
            print(" "*3*indent+str(k)+":")
            print_list(x[k], indent+1, print_values=print_values)
        else:
            if isinstance(x[k], torch.Tensor):
                print(" "*3*indent+str(k)+":", x[k].detach().cpu().numpy().shape, x[k].dtype)
                if print_values:
                    print( x[k][0].detach().cpu().numpy())
            elif isinstance(x[k], np.ndarray):
                print(" "*3*indent+str(k)+":", x[k].shape, x[k].dtype)
                if print_values:
                    print( x[k][0])
            else:
                print(" "*3*indent+str(k)+":", x[k])


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        # required
        root,
        out_dir,
        view,
        # dataset specification
        taskvar=[('close_door', 0)],
    ):
        self._taskvar = taskvar
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._view = view
        # For trajectory optimization, initialize interpolation tools

        # Keep variations and useful instructions
        self._instructions = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if data_dir.is_dir():
                self._num_vars[task] += 1

        # File-names of episodes per task and variation
        episodes_by_task = defaultdict(list)  # {task: [(task, var, filepath)]}
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root / f"{task}+{var}"
            if not data_dir.is_dir():
                print(f"Can't find dataset folder {data_dir}")
                continue
            npy_episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            dat_episodes = [(task, var, ep) for ep in data_dir.glob("*.dat")]
            pkl_episodes = [(task, var, ep) for ep in data_dir.glob("*.pkl")]
            episodes = npy_episodes + dat_episodes + pkl_episodes
            if len(episodes) == 0:
                print(f"Can't find episodes at folder {data_dir}")
                continue
            episodes_by_task[task] += episodes

        # Collect and trim all episodes in the dataset
        self._episodes = []
        self._num_episodes = 0
        for task, eps in episodes_by_task.items():
            episodes_by_task[task] = sorted(
                eps, key=lambda t: int(str(t[2]).split('/')[-1][2:-4])
            )
            self._episodes += eps
            self._num_episodes += len(eps)
        print(f"Created dataset from {root} with {self._num_episodes}")
        self._episodes_by_task = episodes_by_task
        self._out_dir = out_dir


    @staticmethod
    def _unnormalize_rgb(rgb):
        # (from [-1, 1] to [0, 1]) to feed RGB to pre-trained backbone
        return rgb / 2 + 0.5

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id]

        # Load episode
        episode = loader(file)

        # Get the image tensors for the frame ids we got
        states = episode[1] # numpy array

        # Split RGB and XYZ
        pcds = states[:, :, 1]

        t, v, c, h, w = pcds.shape
        pcds= einops.rearrange(pcds, "t v c h w -> (t v h w) c")
        normals = compute_normals(pcds, self._view)
        normals = einops.rearrange(normals, "(t v h w) c -> t v 1 c h w", t=t, v=v, h=h, w=w)
        states = np.concatenate([states, normals], axis=2)
        episode[1] = states

        taskvar_dir = self._out_dir / f"{task}+{variation}"
        # create dir
        taskvar_dir.mkdir(parents=True, exist_ok=True)
        with open(taskvar_dir / f"ep{episode_id}.dat", "wb") as f:
            f.write(blosc.compress(pickle.dumps(episode)))

    def __len__(self):
        return self._num_episodes


class Arguments(tap.Tap):
    data_dir: Path = Path(__file__).parent / "data" / "PeractPackaged"
    tasks: Tuple[str, ...] = ("place_cups", "close_jar", "insert_onto_square_peg", "light_bulb_in", "meat_off_grill", "open_drawer", "place_shape_in_shape_sorter", "place_wine_at_rack_location", "push_buttons", "put_groceries_in_cupboard", "put_item_in_drawer", "put_money_in_safe", "reach_and_drag", "slide_block_to_color_target", "stack_blocks", "stack_cups", "sweep_to_dustpan_of_size", "turn_tap")
    variations: Tuple[int, ...] = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199)
    instructions: Path = Path(__file__).parent / "data" / "peract" / "instructions.pkl"
    output: Path = Path(__file__).parent / "data"
    gripper_path: Path = Path(__file__).parent / "18_peract_tasks_location_bounds.json"
if __name__ == "__main__":
    args = Arguments().parse_args()
    bounds = get_gripper_loc_bounds(args.gripper_path)
    bounds = np.array(bounds)
    view = np.array([0.0, 0.0, 0.0])
    view[:2] = np.mean(bounds[:,:2], axis=0)
    view[2] = bounds[1,2] + 0.5 * (bounds[1,2] - bounds[0,2])
    instruction = load_instructions(
        args.instructions,
        tasks=args.tasks,
        variations=args.variations
    )
    taskvar = [
        (task, var)
        for task, var_instr in instruction.items()
        for var in var_instr.keys()
    ]
    dataset = Dataset(root=args.data_dir, taskvar=taskvar, out_dir=args.output, view=view)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=lambda x: x
    )

    for _ in tqdm(dataloader):
        continue
