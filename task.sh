#!/bin/zsh

source /home/neolux/.zshrc

conda activate mmseg1
source ../envs.tmpfs

python tools/train.py "$@"
