#!/bin/bash

cmd="conda deactivate"
echo ${cmd}
eval ${cmd}
cmd="conda create --force -y -n openss python==3.8"
echo ${cmd}
eval ${cmd}

cmd="conda activate openss"
echo ${cmd}
eval ${cmd}

cmd="conda install pytorch torchvision -c pytorch"
echo ${cmd}
eval ${cmd}

cmd="pip install mmcv==1.4.0"
echo ${cmd}
eval ${cmd}

cmd="pip install -r requirement.txt"
echo ${cmd}
eval ${cmd}

cmd="pip install mmsegmentation mmdet"
echo ${cmd}
eval ${cmd}
