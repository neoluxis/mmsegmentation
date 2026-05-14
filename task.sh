#!/bin/zsh

source /home/neolux/.zshrc

conda activate mmseg1 || conda activate mmseg
source ../envs.tmpfs

python tools/train.py "$@"
