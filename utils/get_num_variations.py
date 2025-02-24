import tap
import os
import numpy as np

class Arguments(tap.Tap):
    dir: str = "/home/share/3D_attn_felix/Peract_packaged/train/"


def main(args):
    files = os.listdir(args.dir)
    tasks = {}
    for file in files:
        task, var = file.split("+")
        if task not in tasks:
            tasks[task] = 0
        tasks[task] += 1

    for task in tasks:
        print(f"{task}: {tasks[task]}")


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args)  
