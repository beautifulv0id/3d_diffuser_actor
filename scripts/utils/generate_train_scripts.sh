
dataset="/home/share/3D_attn_felix/Peract_packaged/train/"
valset="/home/share/3D_attn_felix/Peract_packaged/val/"
instructions="/home/share/3D_attn_felix/rlbench_instructions/instructions.pkl"
gripper_loc_bounds="tasks/18_peract_tasks_location_bounds.json"
tasks="place_cups close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap"
volumes="~/3d_diffuser_actor:/workspace \
    ~/pointattention/:/pointattention \
    /home/share/3D_attn_felix/Peract_packaged/:/workspace/data/Peract_packaged/ \
    /home/share/3D_attn_felix/peract/instructions.pkl:/workspace/data/peract/instructions.pkl"

main_python_scripts=$(ls main_*)
mkdir -p scripts/train
mkdir -p scripts/slurm
for python_script in $main_python_scripts; do
    bash_script="${python_script/main_/train_}"
    bash_script="${bash_script/.py/.sh}"
    python utils/generate_train_script.py $python_script \
        --output_dir scripts \
        --output $bash_script \
        --dataset $dataset \
        --valset $valset \
        --instructions $instructions \
        --gripper_loc_bounds $gripper_loc_bounds \
        --tasks $tasks \
        --volumes $volumes
done
