
if [ "all" == "$1" ]; then
    main_python_scripts=$(ls main_*)
else
    main_python_scripts=$1
fi

mkdir -p scripts/train
mkdir -p scripts/slurm
for python_script in $main_python_scripts; do
    bash_script="${python_script/main_/train_}"
    bash_script="${bash_script/.py/.sh}"
    python utils/generate_train_script.py $python_script \
        --output_dir scripts \
        --output $bash_script 
done
