tasks=(
    close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
)
seeds=(6)
num_tasks=${#tasks[@]}
for seed in ${seeds[@]}; do
    for ((i=0; i<$num_tasks; i++)); do
    task=${tasks[$i]}
        echo "Running $task with seed $seed"
        sbatch slurm_scripts/run_docker.sh slurm_scripts/eval_3dda.sh $task $seed
    done
done