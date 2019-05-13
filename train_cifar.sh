
#/bin/bash

# WHATEVER IN IFCONFIG (like eth0 or something
export GLOO_SOCKET_IFNAME=enp0s31f6

world_size=3
batch_size=128
dataset=cifar

python cifar_resnet.py \
    --rank 1 \
    --world_size $world_size \
    --ip 219.224.168.78 \
    --port 22000 \
    --batch_size $batch_size \
    --epochs 50 \
    --log log/log_${dataset}_node${world_size}_batch${batch_size}.txt
