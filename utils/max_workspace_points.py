import torch
from datasets.dataset_engine import RLBenchDataset
from typing import Tuple, Optional
from pathlib import Path
import tap
import torch.nn.functional as F
import einops


class Arguments(tap.Tap):
    output: Path = Path("/home/stud_herrmann/3d_diffuser_actor/tasks/max_workspace_points.json")
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder", "front")
    image_size: str = "256,256"
    max_episodes_per_task: int = 100
    instructions: Optional[Path] = "/home/stud_herrmann/3d_diffuser_actor/data/peract/instructions.pkl"
    tasks: Tuple[str, ...] = ("place_cups","close_jar","insert_onto_square_peg","light_bulb_in","meat_off_grill","open_drawer","place_shape_in_shape_sorter","place_wine_at_rack_location","push_buttons","put_groceries_in_cupboard","put_item_in_drawer","put_money_in_safe","reach_and_drag","slide_block_to_color_target","stack_blocks","stack_cups","sweep_to_dustpan_of_size","turn_tap"),
    variations: Tuple[int, ...] = (*range(1, 199),)
    buffer = 0.2
    min_workspace = torch.tensor([-0.8, -0.8,  0.5])
    max_workspace = torch.tensor([1.2, 0.7, 1.8])

    # Training and validation datasets
    dataset: Path = "/home/share/3D_attn_felix/peract/Peract_packaged_normals/train"
    valset: Path = "/home/share/3D_attn_felix/peract/Peract_packaged_normals/val"
    dense_interpolation: int = 1
    interpolation_length: int = 2

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = "train_logs"
    exp_log_dir: str = "exp"
    run_log_dir: str = "run"
    name: str = 'train_3d_diffuser_actor'

    # Main training parameters
    num_workers: int = 1
    batch_size: int = 16
    batch_size_val: int = 4
    cache_size: int = 100
    max_episode_length: int = 5  # -1 for no limit

    # Data augmentations
    rot_noise: float = 0.0
    pos_noise: float = 0.0
    pcd_noise: float = 0.0
    image_rescale: str = "0.75,1.25"  # (min, max), "1.0,1.0" for no rescaling




def get_datasets(args):
    """Initialize datasets."""
    taskvar = [
        (task, var)
        for task in args.tasks
        for var in range(0, 199)
    ]
    print(taskvar)

    # Initialize datasets with arguments
    train_dataset = RLBenchDataset(
        root=args.dataset,
        instructions=None,
        taskvar=taskvar,
        max_episode_length=args.max_episode_length,
        cache_size=args.cache_size,
        max_episodes_per_task=args.max_episodes_per_task,
        num_iters=None,
        cameras=args.cameras,
        training=True,
        image_rescale=tuple(
            float(x) for x in args.image_rescale.split(",")
        ),
        return_low_lvl_trajectory=True,
        dense_interpolation=bool(args.dense_interpolation),
        interpolation_length=args.interpolation_length,
        rot_noise=args.rot_noise,
        pos_noise=args.pos_noise,
        pcd_noise=args.pcd_noise,
    )
    return train_dataset

def traj_collate_fn(batch):
    keys = [
        "trajectory", "trajectory_mask",
        "rgbs", "pcds",
        "curr_gripper", "curr_gripper_history", "action", "instr"
    ]
    ret_dict = {
        key: torch.cat([
            item[key].float() if key != 'trajectory_mask' else item[key]
            for item in batch
        ]) for key in keys
    }

    ret_dict["task"] = []
    for item in batch:
        ret_dict["task"] += item['task']
    return ret_dict

import torch

def get_max_points(points, tasks, min_bound, max_bound):
    """
    Crop point clouds to a 3D workspace and resample to a fixed size.
    
    Args:
        points (torch.Tensor): Input batch of point clouds. Shape: (B, N, 3)
        min_bound (torch.Tensor): Minimum bounds of workspace. Shape: (3,)
        max_bound (torch.Tensor): Maximum bounds of workspace. Shape: (3,)
    
    Returns:
        torch.Tensor: Processed point clouds with shape (B, max_points, 3)
    """
    device = points.device
    B, N, _ = points.shape
    
    # Create mask for points within workspace
    mask = (points >= min_bound) & (points <= max_bound)
    mask = mask.all(dim=-1)  # (B, N)
    

    max_points = {
        task: 0
        for task in tasks
    }
    
    for i, task in enumerate(tasks):
        # Extract valid points for this cloud
        valid_points = points[i][mask[i]]  # (K, 3)
        K = valid_points.size(0)
        
        if K == 0:
            raise ValueError(f"All points filtered in cloud {i}. Consider checking bounds or input data.")
        
        # Find the maximum number of points in this batch
        max_points[task] = max(max_points[task], K)

    return max_points

def crop(points, rgbs, min_bound, max_bound):
    """
    Crop point clouds to a 3D workspace and resample to a fixed size.
    
    Args:
        points (torch.Tensor): Input batch of point clouds. Shape: (B, N, 3)
        min_bound (torch.Tensor): Minimum bounds of workspace. Shape: (3,)
        max_bound (torch.Tensor): Maximum bounds of workspace. Shape: (3,)
        max_points (int): Target number of points per cloud after processing
    
    Returns:
        torch.Tensor: Processed point clouds with shape (B, max_points, 3)
    """
    device = points.device
    B, N, _ = points.shape
    
    # Create mask for points within workspace
    mask = (points >= min_bound) & (points <= max_bound)
    mask = mask.all(dim=-1)  # (B, N)
    
    pcd = []
    rgb = []
    
    for i in range(B):
        # Extract valid points for this cloud
        valid_points = points[i][mask[i]]  # (K, 3)
        pcd.append(valid_points)
        rgb.append(rgbs[i][mask[i]])
    
    return pcd, rgb

def interpolate(pcds, feat_h=32, feat_w=32):
    ncam = pcds.shape[1]
    pcds = einops.rearrange(pcds, "bt ncam c h w -> (bt ncam) c h w")
    pcds = F.interpolate(
        pcds,
        (feat_h, feat_w),
        mode='bilinear'
    )
    pcds = einops.rearrange(
        pcds,
        "(bt ncam) c h w -> bt ncam c h w", ncam=ncam
    )
    return pcds


if __name__ == "__main__":
    args = Arguments().parse_args()
    train_dataset = get_datasets(args)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=traj_collate_fn
    )

    max_points = {task: 0 for task
                  in args.tasks}

    batch = next(iter(loader))
    b = batch['pcds'].shape[0]
    task = batch['task'][0]
    pcds = interpolate(batch['pcds'], 32, 32).permute(0, 1, 3, 4, 2)[0:1].reshape(1, -1, 3)
    rgbs = interpolate(batch['rgbs'], 32, 32).permute(0, 1, 3, 4, 2)[0:1].reshape(1, -1, 3)
    pcds, rgbs = crop(pcds, rgbs, args.min_workspace, args.max_workspace)
    pcds = pcds[0].cpu().numpy().reshape(-1, 3)
    rgbs = rgbs[0].cpu().numpy().reshape(-1, 3)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcds[:, 0], pcds[:, 1], pcds[:, 2], c=rgbs, s=1)
    plt.savefig(Path(args.output).parent / "imgs" / f"{task}.png")
    plt.close()


    # for batch in tqdm.tqdm(loader):
    #     b = batch['pcds'].shape[0]
    #     pcds = batch['pcds']
    #     pcds = interpolate(pcds, 32, 32)
    #     pcds = pcds.permute(0, 1, 3, 4, 2).reshape(b, -1, 3)
    #     tasks = batch['task']
    #     this_max_points = get_max_points(pcds, tasks, args.min_workspace, args.max_workspace)
    #     for task, max_point in this_max_points.items():
    #         max_points[task] = max(max_points[task], max_point)

    # if os.path.exists(args.output):
    #     with open(args.output, 'r') as f:
    #         existing_max_points = json.load(f)
    # else:
    #     existing_max_points = {}

    # existing_max_points.update(max_points)
    # max_points = existing_max_points

    # with open(args.output, 'w') as f:
    #     json.dump(max_points, f, indent=4)