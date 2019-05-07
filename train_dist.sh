#/bin/bash

# WHATEVER IN IFCONFIG (like eth0 or something  
export GLOO_SOCKET_IFNAME=enp3s0

python train_dist.py \
    --rank 0 \
    --world_size 2 \
    --ip 219.224.168.78 \
    --port 22000 \
    #--batch_size 64 \
    #--epochs 10 
