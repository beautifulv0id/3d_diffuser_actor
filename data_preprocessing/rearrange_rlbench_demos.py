import os
from subprocess import call
import pickle
from pathlib import Path

import tap


class Arguments(tap.Tap):
    root_dir: Path


def main(root_dir, task):
    variations = os.listdir(f'{root_dir}/{task}/all_variations/episodes')
    seen_variations = {}
    for variation in variations:
        num = int(variation.replace('episode', ''))
        variation = pickle.load(
            open(
                f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_number.pkl',
                'rb'
            )
        )

        if variation not in seen_variations.keys():
            seen_variations[variation] = [num]
            dir_path = f'{root_dir}/{task}/variation{variation}'
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                try:
                    call(["rm", "-rf", dir_path])  # Linux/Mac
                    print(f"Directory '{dir_path}' has been deleted.")
                except Exception as e:
                    print(f"Error deleting directory: {e}")
        else:
            seen_variations[variation].append(num)

        os.makedirs(f'{root_dir}/{task}/variation{variation}/episodes', exist_ok=True)

        if os.path.isfile(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl'):
            data1 = pickle.load(open(f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl', 'rb'))
            data2 = pickle.load(open(f'{root_dir}/{task}/variation{variation}/variation_descriptions.pkl', 'rb'))
            assert data1 == data2
        else:
            call(['ln', '-s',
                  f'{root_dir}/{task}/all_variations/episodes/episode{num}/variation_descriptions.pkl',
                  f'{root_dir}/{task}/variation{variation}/'])

        ep_id = len(seen_variations[variation]) - 1
        call(['ln', '-s',
              "{:s}/{:s}/all_variations/episodes/episode{:d}".format(root_dir, task, num),
              f'{root_dir}/{task}/variation{variation}/episodes/episode{ep_id}'])


if __name__ == '__main__':
    args = Arguments().parse_args()
    root_dir = str(args.root_dir.absolute())
    tasks = [f for f in os.listdir(root_dir) if '.zip' not in f]
    for task in tasks:
        main(root_dir, task)
