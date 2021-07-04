#!/bin/bash

# Setup environment (do not change this)
source activate pytorch_p36
pip install -r requirements.txt
export WANDB_API_KEY=$(cat "aws_configs/wandb.key")

# Download dataset (do not change this)
if [ ! -d "/home/ubuntu/miniscapes" ]; then
  echo "Download miniscapes"
  aws s3 cp s3://dlad-miniscapes-2021/miniscapes.zip /home/ubuntu/
  echo "Extract miniscapes"
  unzip /home/ubuntu/miniscapes.zip -d /home/ubuntu/ | awk 'BEGIN {ORS=" "} {if(NR%1000==0)print "."}'
  rm /home/ubuntu/miniscapes.zip
  echo "\n"
fi

#python -m mtl.scripts.compute_statistics miniscapes /home/ubuntu/miniscapes/
# Run training
echo "Start training"
cd /home/ubuntu/code/

# BEGIN YOUR CHANGES HERE
# You can specify the hyperparameters and the experiment name here.
python -m mtl.scripts.train \
  --log_dir /home/ubuntu/results/ \
  --dataset_root /home/ubuntu/miniscapes/ \
  --name 3_adam_0001_distillation \
  --optimizer adam \
  --optimizer_lr 0.0001 \
  --loss_weight_semseg 0.6 \
  --loss_weight_depth 0.4 \
  --model_name deeplabv3pdistillation \
  --batch_size 4 \
  --num_epochs 16 \
  #--aug_geom_scale_min 0.7 \
  #--aug_geom_scale_max 1.3 \
  #--resume s3://yunke.1/G4_0505-2319_4_adam_lr_00002_weight_sd_64_depthlog_2up_skip2x_rase_full_2db91/
# ... you can pass further arguments as specified in utils/config.py

# If you want to run multiple experiments after each other, just call the training script multiple times.
# Don't forget to check if the AWS timeout in aws_start_instances.py is still sufficient.
#  python -m mtl.scripts.train \
#    --log_dir /home/ubuntu/results/ \
#    --dataset_root /home/ubuntu/miniscapes/ \
#    --name Default
#python -m mtl.scripts.train \
#  --log_dir /home/ubuntu/results/ \
#  --dataset_root /home/ubuntu/miniscapes/ \
#  --name adam_lr_0001_weight_sd_64 \
#  --optimizer adam \
#  --optimizer_lr 0.0001 \
#  --loss_weight_semseg 0.6 \
#  --loss_weight_depth 0.4

# END YOUR CHANGES HERE

# Wait a moment before stopping the instance to give a chance to debug
echo "Terminate instance in 2 minutes. Use Ctrl+C to cancel the termination..."
sleep 2m && bash aws_stop_self.sh
