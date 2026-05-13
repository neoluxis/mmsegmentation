#!/bin/zsh

source /home/neolux/.zshrc

conda activate mmseg1
source ../envs 

for exp in work_dirs/*; do
    python tools/analysis_tools/plot_scalars.py $exp
done

