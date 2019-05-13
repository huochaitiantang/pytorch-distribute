
#/bin/bash

# WHATEVER IN IFCONFIG (like eth0 or something
export GLOO_SOCKET_IFNAME=eth1

world_size=1
batch_size=128
dataset=cifar

python cifar_resnet.py \
    --rank 0 \
    --world_size $world_size \
    --ip 10.1.192.195 \
    --port 22000 \
    --batch_size $batch_size \
    --epochs 1\
    --log log/log_${dataset}_node${world_size}_batch${batch_size}.txt
