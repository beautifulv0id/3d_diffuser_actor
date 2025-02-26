
if [ "all" == "$1" ]; then
    main_python_scripts=$(ls main_*)
else
    main_python_scripts=$1
fi

mkdir -p scripts/train
mkdir -p scripts/slurm
for python_script in $main_python_scripts; do
    name="${python_script/main_/}"
    name="${name/.py/}"
    python utils/generate_eval_script.py $name 
done
