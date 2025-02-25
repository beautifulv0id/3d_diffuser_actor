tasks=(
    close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
)
split=val
num_ckpts=${#tasks[@]}
for ((i=0; i<$num_ckpts; i++)); do
echo ${tasks[$i]}
python data_preprocessing/add_normals_to_packaged.py --tasks ${tasks[$i]} --data_dir /data/Peract_packaged/$split/ --instructions /data/peract/instructions.pkl --output /data/Peract_packaged_normals_val/$split --gripper_path tasks/18_peract_tasks_location_bounds.json &
done