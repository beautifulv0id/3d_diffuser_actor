import wandb
import pandas as pd
from tqdm import tqdm

# Configuration
WANDB_PROJECT = "3d_diffuser_actor_debug"  # Replace with your project name
RUN_ID_1 = "xlbbbu8w"  # Replace with the first run's ID
RUN_ID_2 = "ynund3pn"  # Replace with the second run's ID
MERGED_RUN_NAME = "pointattnlangenhanced_multitask-C120-B24-lr1e-4-DI1-2-H3-DT100"

# Initialize wandb API
api = wandb.Api()

# Fetch runs
run1 = api.run(f"{WANDB_PROJECT}/{RUN_ID_1}")
run2 = api.run(f"{WANDB_PROJECT}/{RUN_ID_2}")

# Get histories as DataFrames
history1 = pd.DataFrame(run1.history())
history2 = pd.DataFrame(run2.history())

# Handle any NaN timestamps or metrics by dropping empty rows
history1.dropna(how="all", inplace=True)
history2.dropna(how="all", inplace=True)

# Merge the histories
# Optionally, you can adjust or sort based on specific conditions
merged_history = pd.concat([history1, history2], ignore_index=True)

# Start a new wandb run
with wandb.init(project=WANDB_PROJECT, name=MERGED_RUN_NAME) as new_run:
    for _, row in tqdm(merged_history.iterrows(), total=len(merged_history)):
        log_dict = row.dropna().to_dict()  # Drop NaNs before logging
        wandb.log(log_dict)

print("Merged run completed.")
