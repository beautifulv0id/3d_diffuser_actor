import json
import tap
import os
import numpy as np

class Arguments(tap.Tap):
    dir: str = "eval_logs/3dda/2025.01.27/07.40.39_Actor_18Peract_100Demo_multitask/diffusion_multitask-C120-B16-lr1e-4-DI1-2-H3-DT100"

def extract_values(args):
    values = {}
    for seed in os.listdir(args.dir):
        for file in os.listdir(os.path.join(args.dir, seed)):
            if file.endswith(".json"):
                task = file.split(".")[0]
                if task not in values:
                    values[task] = {}
                    values[task]['vals'] = []
                with open(os.path.join(args.dir, seed, file)) as f:
                    data = json.load(f)
                    values[task]['vals'].append(data[task]['mean'])
    for task in values:
        values[task]['mean'] = np.mean(values[task]['vals'])
        values[task]['std'] = np.std(values[task]['vals'])
    return values


if __name__ == "__main__":
    args = Arguments().parse_args()
    values = extract_values(args)
    for task in values:
        print(f"{task}: {values[task]['mean']} +/- {values[task]['std']}")

    with open(os.path.join(args.dir, "summary.json"), "w") as f:
        json.dump(values, f, indent=4)

    