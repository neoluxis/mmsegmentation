#!/bin/zsh

source /home/neolux/.zshrc

conda activate mmseg1
source ../envs 

python tools/train.py $1

