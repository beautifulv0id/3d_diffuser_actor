import wandb
import pickle
import os
import tqdm

def get_sorted_runs(project_name, entity=None, metric="val-losses/mean/pos_l2_final<0.01", cached_data={}):    
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}" if entity else project_name)
    
    run_data = []
    
    for run in tqdm.tqdm(runs):
        if not(len(run.config['tasks']) == 1 and run.config['tasks'][0] == 'stack_blocks'):
            continue
        
        if run.id in cached_data:
            metric_values = cached_data[run.id][metric]
        else:
            history = run.scan_history(keys=[metric], page_size=200000,)
            metric_values = [row.get(metric, 0) for row in history]
            cached_data[run.id] = {}
            cached_data[run.id][metric] = metric_values
            cached_data[run.id]["config"] = run.config
            cached_data[run.id]["name"] = run.config['name']
            cached_data[run.id]["metadata"] = run.metadata
            
        max_metric_value = max(metric_values) if metric_values else float('-inf')
        run_data.append((run, max_metric_value))
    
    with open(cache_file, "wb") as f:
        pickle.dump(cached_data, f)
    
    sorted_runs = sorted(run_data, key=lambda x: x[1], reverse=True)
    
    return sorted_runs

def get_config(run, cached_data={}):
    if run.id in cached_data:
        return cached_data[run.id]["config"]
    else:
        cached_data[run.id] = {
            "config": run.config
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f)
        return cached_data[run.id]["config"]

def get_gpu_count(run, cached_data={}):
    if run.id in cached_data:
        return cached_data[run.id]["metadata"].get("gpu_count", 1)
    else:
        gpu_count = run.metadata.get("gpu_count", 1)
        cached_data[run.id] = {
            "metadata": {
                "gpu_count": gpu_count
            }
        }
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f)
        return gpu_count    
if __name__ == "__main__":
    project = "3d_diffuser_actor_debug"  # Change this
    entity = "felix-herrmann"  # Change this if applicable
    # run_ids = ["toyzo1d5"]  # Define specific run IDs to include
    run_ids = None  # Include all runs
    metric = "val-losses/mean/pos_l2_final<0.01"  # Change this
    cache_file = "wandb_cache.pkl"  # Change this if needed
    
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
    else:
        cached_data = {}

    
    sorted_results = get_sorted_runs(project, entity, cached_data=cached_data)
    
    for run, value in sorted_results:
        config = get_config(run, cached_data=cached_data)
        gpu_count = get_gpu_count(run, cached_data=cached_data)

        B = config.get('batch_size', 1) * gpu_count
        RN = config.get('rot_noise', 0)
        PN = config.get('pos_noise', 0)
        PCDN = config.get('pcd_noise', 0)
        D = config.get('dropout', 0)
        RES = config.get('feature_res', 'res3')
        FPS = config.get('fps_subsampling_factor', 0)
        HP = config.get('gripper_history_as_points', 0)
        CD = config.get('use_center_distance', 1)
        CP = config.get('use_center_projection', 1)
        VP = config.get('use_vector_projection', 1)
        AC = config.get('add_center', 1)

        print(f"Max : {value},\t B={B:2}, RN={RN:.2f}, PN={PN:.2f}, PCDN={PCDN:.2f}, D={D}, RES={RES}, FPS={FPS:2}, HP={HP}, CD={CD}, CP={CP}, VP={VP}, AC={AC}, Run ID: {run.id}, Name: {run.config['name']}")
