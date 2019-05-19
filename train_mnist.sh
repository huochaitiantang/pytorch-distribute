#/bin/bash

# WHATEVER IN IFCONFIG (like eth0 or something  
export GLOO_SOCKET_IFNAME=enp3s0

world_size=1
batch_size=96
dataset=mnist
rank=0

python train_dist.py \
    --rank $rank \
    --world_size $world_size \
    --ip 219.224.168.78 \
    --port 22000 \
    --batch_size $batch_size \
    --epochs 10 \
    --log log/log_${dataset}_world${world_size}_rank${rank}_batch${batch_size}.txt
