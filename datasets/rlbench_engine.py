import itertools
import numpy as np
import torch
import zarr
from collections import defaultdict, Counter
from .utils import Resize, dict_apply
from pathlib import Path
import random
from time import time

def create_sample_indices(split, taskvar, n_episodes, n_obs):
    indices = []
    for (task, var) in taskvar:
        taskvar_group = split[task][str(var)]
        for i, demo_group in enumerate(taskvar_group.values()):
            trajectory_length = demo_group['state_action']['proprioception'].shape[0]
            for action_idx in range(1, trajectory_length):
                obs_idxs = []
                for offset in reversed(range(1, n_obs+1)):
                    if action_idx - offset < 0:
                        obs_idxs.append(0)
                    else:
                        obs_idxs.append(action_idx-offset)
                indices.append({
                    'task': task,
                    'var': var,
                    'demo': demo_group,
                    'obs_idxs': obs_idxs,
                    'action_idx': action_idx
                })
    if n_episodes > 0:
        indices = random.sample(indices, n_episodes)
    return indices

def collate_samples(datum, instructions, apply_cameras):
    sample = {
        'obs': dict(),
        'action': dict()
    }

    obs_idxs = datum['obs_idxs']
    next_keypoint_idx = datum['action_idx']
    cameras = datum['demo']['cameras']
    state_action = datum['demo']['state_action']

    sample['pcds'] = np.stack([cameras[camera]['pcd'][obs_idxs[-1]] for camera in apply_cameras])
    rgb = np.stack([cameras[camera]['rgb'][obs_idxs[-1]] for camera in apply_cameras])
    rgb = rgb.astype(np.float32) / 255.0
    sample['rgbs'] = rgb

    sample['curr_gripper'] = state_action['proprioception'][:][obs_idxs]
    sample['trajectory'] = state_action['proprioception'][next_keypoint_idx].reshape(1, -1)

    task = datum['task']
    var = datum['var']

    if instructions:
        instr = random.choice(instructions[task][var])
        instr = instr[None].repeat(1, 1, 1)
    else:
        instr = torch.zeros((1, 53, 512))

    sample['instr'] = instr
    sample['task'] = task

    return sample

class RLBenchDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 root: str,
                 instructions = None,
                 cameras = ['left_shoulder', 'right_shoulder', 'wrist', 'front'],
                 taskvar = [('open_drawer', 0)],
                 n_obs_steps = 3,
                 n_episodes = 1,
                 image_rescale=(1.0, 1.0),
                 cache_size=0,
                 split='train',
                 ):
        
        self._training = True if split == 'train' else False

        if self._training:
            self._resize = Resize(scales=image_rescale)

        root = Path(root)
        split_path = root / split

        print(f"Loading dataset from {split_path}")
        print("Cache size: ", cache_size)

        # Keep variations and useful instructions
        self._instructions = defaultdict(dict)
        self._num_vars = Counter()  # variations of the same task
        this_taskvar = []
        for root_, (task, var) in itertools.product([split_path], taskvar):
            data_dir = root_ / task / str(var)
            if data_dir.is_dir():
                if instructions is not None:
                    self._instructions[task][var] = instructions[task][var]
                self._num_vars[task] += 1
                this_taskvar.append((task, var))


        # read from zarr dataset
        split_root = zarr.open(split_path, 'r')
        indices = create_sample_indices(split_root, this_taskvar, n_episodes, n_obs_steps)

        self.indices = indices
        self.cameras = cameras
        self._cache = dict()
        self._cache_size = cache_size
        self.split = split
        self._root = root
        self.taskvar = taskvar
        self.n_obs_steps = n_obs_steps
        self.n_episodes = n_episodes
        self.image_rescale = image_rescale
        self.cache_size = cache_size

        print(f"Loaded {len(self)} {split} samples")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):      
        if idx in self._cache:
            sample = self._cache[idx]
        else:
            index = self.indices[idx]
            sample = collate_samples(
                index,
                self._instructions,
                apply_cameras=self.cameras,
            )

            sample = dict_apply(sample, lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x)

            if len(self._cache) == self._cache_size and self._cache_size > 0:
                key = list(self._cache.keys())[int(time()) % self._cache_size]
                del self._cache[key]

            if len(self._cache) < self._cache_size:
                self._cache[idx] = sample

        if self._training:
            sample.update(self._resize(rgb=sample['rgbs'], pcd=sample['pcds']))

        return sample
    
    def get_dataset(self, split):
        dataset = RLBenchDataset(
            root=self._root,
            cameras=self.cameras,
            taskvar=self.taskvar,
            n_obs_steps=self.n_obs_steps,
            n_episodes=self.n_episodes,
            image_rescale=self.image_rescale,
            cache_size=self.cache_size,
            split=split,
        )
        return dataset

    def get_test_dataset(self):
        dataset = self.get_dataset('test')
        dataset._training = False
        return dataset

    def get_validation_dataset(self):
        dataset = self.get_dataset('val')
        dataset._training = False
        return dataset
    
    def empty_cache(self):
        for k, v in self._cache.items():
            del v
        self._cache = dict()
