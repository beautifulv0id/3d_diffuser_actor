import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
import copy
import json

def get_all_files_and_other_directories(mypath):
    mypath = mypath + '/'
    # list all files in current directory:
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    onlydirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]

    for i in range(len(onlyfiles)):
        onlyfiles[i] = join(mypath, onlyfiles[i])
    for i in range(len(onlydirs)):
        onlydirs[i] = join(mypath, onlydirs[i])

    if len(onlydirs)!=0:
        for i in range(len(onlydirs)):
            # print (onlydirs[i])
            onlyfiles_more = get_all_files_and_other_directories(copy.deepcopy(onlydirs[i]))
            onlyfiles.extend(onlyfiles_more)

    return onlyfiles

def filter_file_names(path_list, filter="eval"):
    ret_list = []
    for i in range(len(path_list)):
        if (path_list[i][-5:]==".json"):
            if (filter in path_list[i].split("/")[-1]):
                ret_list.append(path_list[i])
    return ret_list

def compute_metrics(json_files):
    task_name = []
    succ_metrics_list = []
    for i in range(len(json_files)):
        with open(json_files[i]) as data_file:
            data = json.load(data_file)
            # print (data)
            for key, value in data.items():
                task_name.append(key)
                succ_metrics_list.append(value['mean'])
    return np.mean(np.asarray(succ_metrics_list)), np.std(np.asarray(succ_metrics_list)), task_name[0]

def do_eval(path):
    all_files = get_all_files_and_other_directories(path)
    eval_files = filter_file_names(all_files)
    mean, std, task_name = compute_metrics(eval_files)
    return mean, std, task_name


task_names = ["insert_onto_square_peg", "place_cups", "place_shape_in_shape_sorter", "stack_blocks", "stack_cups"]

for i in range(len(task_names)):
    curr_task_name = task_names[i]

    # directory1 = "/media/funk/INTENSO/300_RL_BENCH/0_evals/final_trains_29_01_evening/insert_onto_square_peg/3dda"
    directory1 = "/media/funk/INTENSO/300_RL_BENCH/0_evals/final_trains_29_01_evening/" + curr_task_name + "/3dda"
    name1 = "3dda"
    directory2 = "/media/funk/INTENSO/300_RL_BENCH/0_evals/final_trains_29_01_evening/insert_onto_square_peg/ursa"
    directory2 = "/media/funk/INTENSO/300_RL_BENCH/0_evals/final_trains_29_01_evening/" + curr_task_name + "/ursa"
    name2 = "ursa"

    print (name1)
    print (do_eval(directory1))

    print (name2)
    print (do_eval(directory2))








