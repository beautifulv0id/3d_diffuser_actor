"""Online evaluation script on RLBench."""
import random
from typing import Tuple, Optional
from pathlib import Path
import json
import os

import torch
import numpy as np
import tap

import json

# model imports
from action_flow.se3_flow_matching import SE3FlowMatching
from action_flow.se3_flow_matching_lang_enhanced_nursa_sa import SE3FlowMatchingNURSASA
from action_flow.se3_flow_matching_lang_enhanced import SE3FlowMatchingLangEnhanced
from action_flow.se3_flow_matching_lang_enhanced_sa import SE3FlowMatchingLangEnhancedSA
from action_flow.se3_flow_matching_lang_enhanced_ipa import SE3FlowMatchingLangEnhancedIPA
from action_flow.se3_flow_matching_self_attn import SE3FlowMatchingSelfAttn
from action_flow.se3_flow_matching_lang_enhanced_ipa_sa import SE3FlowMatchingLangEnhancedIPASA
from action_flow.se3_flow_matching_super_point_encoder import SE3FlowMatchingSuperPointEncoder

from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from diffuser_actor.trajectory_optimization.diffuser_actor_flow_matching import DiffuserActorFlowMatching
from diffuser_actor.trajectory_optimization.diffuser_actor_flow_matching_6D import DiffuserActorFlowMatching6D
from diffuser_actor.trajectory_optimization.diffuser_actor_ipa_sa import DiffuserActorIPASA
from diffuser_actor.trajectory_optimization.diffuser_actor_nursa import DiffuserActorNURSA
from diffuser_actor.trajectory_optimization.diffuser_actor_nursa_sa import DiffuserActorNURSASA
from diffuser_actor.trajectory_optimization.diffuser_actor_nursa_sa_local import DiffuserActorNURSASA
from diffuser_actor.trajectory_optimization.diffuser_actor_ursa import DiffuserActorURSA
from diffuser_actor.trajectory_optimization.diffuser_actor_wo_sa import DiffuserActorWoSA
from diffuser_actor.trajectory_optimization.diffuser_actor_wo_sa_nursa import DiffuserActorWoSANURSA
from diffuser_actor.trajectory_optimization.diffuser_actor_wo_sa_ursa import DiffuserActorWoSAURSA
from diffuser_actor.trajectory_optimization.diffuser_actor_wo_sa_ursa_flow_matching import DiffuserActorWoSAURSA
from diffuser_actor.trajectory_optimization.diffuser_actor_wo_sa_ursa_flow_matching_6D import DiffuserActorWoSAURSA6D
from diffuser_actor.trajectory_optimization.diffuser_actor_wo_sa_ursa_local import DiffuserActorWoSAURSALocal

from utils.common_utils import (
    load_instructions,
    get_gripper_loc_bounds,
    round_floats
)
from utils.utils_with_rlbench import RLBenchEnv, Actioner, load_episodes
import inspect

model_class_dict = {
    "pointattn": SE3FlowMatching,
    "pointattn_lang_enhanced_nursa_sa": SE3FlowMatchingNURSASA,
    "pointattn_lang_enhanced": SE3FlowMatchingLangEnhanced,
    "pointattn_lang_enhanced_sa": SE3FlowMatchingLangEnhancedSA,
    "pointattn_lang_enhanced_ipa": SE3FlowMatchingLangEnhancedIPA,
    "pointattn_self_attn": SE3FlowMatchingSelfAttn,
    "pointattn_lang_enhanced_ipa_sa": SE3FlowMatchingLangEnhancedIPASA,
    "pointattn_super_point_encoder": SE3FlowMatchingSuperPointEncoder,    
    "3d_diffuser_actor": DiffuserActor,
    "3d_diffuser_actor_flow_matching": DiffuserActorFlowMatching,
    "3d_diffuser_actor_flow_matching_6D": DiffuserActorFlowMatching6D,
    "3d_diffuser_actor_ipa_sa": DiffuserActorIPASA,
    "3d_diffuser_actor_nursa": DiffuserActorNURSA,
    "3d_diffuser_actor_nursa_sa": DiffuserActorNURSASA,
    "3d_diffuser_actor_nursa_sa_local": DiffuserActorNURSASA,
    "3d_diffuser_actor_ursa": DiffuserActorURSA,
    "3d_diffuser_actor_wo_sa": DiffuserActorWoSA,
    "3d_diffuser_actor_wo_sa_nursa": DiffuserActorWoSANURSA,
    "3d_diffuser_actor_wo_sa_ursa": DiffuserActorWoSAURSA,
    "3d_diffuser_actor_wo_sa_ursa_flow_matching": DiffuserActorWoSAURSA,
    "3d_diffuser_actor_wo_sa_ursa_flow_matching_6D": DiffuserActorWoSAURSA6D,
    "3d_diffuser_actor_wo_sa_ursa_local": DiffuserActorWoSAURSALocal,
}

class Arguments(tap.Tap):
    checkpoint: Path = ""
    seed: int = 2
    device: str = "cuda"
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = "instructions.pkl"
    variations: Tuple[int, ...] = (-1,)
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    verbose: int = 0
    output_file: Path = Path(__file__).parent / "eval.json"
    max_steps: int = 25
    test_model: str = "3d_diffuser_actor"
    collision_checking: int = 0
    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    gripper_loc_bounds_buffer: float = 0.04
    single_task_gripper_loc_bounds: int = 0
    predict_trajectory: int = 1
    hyper_params_file: Path = Path("hparams.json")
    action_dim: int = 8

def get_signature_params(func):
    sig = inspect.signature(func)
    
    # Get parameter names and default values
    params = {}
    for name, param in sig.parameters.items():
        # Store parameter name with its default value (if any)
        if param.default is not param.empty:
            params[name] = param.default
        else:
            params[name] = None  # No default value

    params.pop('self', None)
    
    return params

def get_class_kwargs(func, kwargs):
    params = get_signature_params(func)
    return {k: v for k, v in kwargs.items() if k in params}


def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    print('Gripper workspace')
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=task, buffer=args.gripper_loc_bounds_buffer,
    )
    args.gripper_loc_bounds = gripper_loc_bounds

    model_class = model_class_dict[args.test_model]
    kwargs = get_class_kwargs(model_class, vars(args))
    model = model_class(
        **kwargs
    ).to(device)

    # Load model weights
    model_dict = torch.load(args.checkpoint, map_location="cpu")
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight)
    model.eval()

    return model

def load_hparams(file):
    with open(file, "r") as f:
        hparams = json.load(f)
    hparams = {k: v for k, v in hparams.items() if not isinstance(v, dict)}
    return hparams

if __name__ == "__main__":
    # Arguments
    args = Arguments().parse_args()
    hparams = load_hparams(args.hyper_params_file)
    for k, v in hparams.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.cameras = tuple(x for y in args.cameras for x in y.split(","))
    args.image_size = tuple(int(x) for x in args.image_size.split(","))
    print("Arguments:")
    print(args)
    print("-" * 100)
    # Save results here
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    model = load_models(args)

    # Load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        image_size=args.image_size,
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking)
    )

    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise NotImplementedError()

    actioner = Actioner(
        policy=model,
        instructions=instruction,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory),
        test_model=args.test_model
    )
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates = {}

    for task_str in args.tasks:
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=(
                max_eps_dict[task_str] if args.max_steps == -1
                else args.max_steps
            ),
            num_variations=args.variations[-1] + 1,
            num_demos=args.num_episodes,
            actioner=actioner,
            max_tries=args.max_tries,
            dense_interpolation=bool(args.dense_interpolation),
            interpolation_length=args.interpolation_length,
            verbose=bool(args.verbose),
            num_history=args.num_history
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(args.output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
