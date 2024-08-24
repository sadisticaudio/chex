#! /bin/bash

# python setup.py build bdist_wheel
# mkdir -p realBuild
# yes | cp -rf build/* realBuild/
# rm -rf build

# eval "$(conda shell.bash hook)"
# conda activate python3.8
# pip install build wheel
# python -m build --wheel
# conda deactivate
# # yes | cp -rf build/* realBuild/
# # rm -rf build

# eval "$(conda shell.bash hook)"
# conda activate python3.9
# pip install build wheel
# python -m build --wheel
# conda deactivate
# # yes | cp -rf build/* realBuild/
# # rm -rf build

# eval "$(conda shell.bash hook)"
# conda activate python3.10
# pip install build wheel
# python -m build --wheel
# conda deactivate
# # yes | cp -rf build/* realBuild/
# # rm -rf build

eval "$(conda shell.bash hook)"
conda activate python3.11
pip install build wheel
python -m build --wheel
conda deactivate
# yes | cp -rf build/* realBuild/
# rm -rf build

# eval "$(conda shell.bash hook)"
# conda activate python3.12
# pip install build wheel
# python -m build --wheel
# conda deactivate
# # yes | cp -rf build/* realBuild/
# # rm -rf build
