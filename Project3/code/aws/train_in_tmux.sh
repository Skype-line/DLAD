#!/bin/bash
tmux new-session -d -s dlad -n train
tmux send-keys -t dlad:train "source activate pytorch_latest_p37" Enter
tmux send-keys -t dlad:train "cd ~/code/utils/Rotated_IoU/cuda_op/" Enter
tmux send-keys -t dlad:train "python setup.py install" Enter
tmux send-keys -t dlad:train "cd ~/code && bash aws/train.sh" Enter
