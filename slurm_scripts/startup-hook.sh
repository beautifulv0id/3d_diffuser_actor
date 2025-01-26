export PERACT_TEST_DIR=/workspace/data/peract/test/
export PERACT_INSTRUCTIONS=/workspace/data/peract/instructions.pkl
source activate 3d_diffuser_actor
cd /pointattention/
pip install -e .
cd /workspace/
pip install -e .
pip install open3d
