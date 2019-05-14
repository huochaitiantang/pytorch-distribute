
#/bin/bash

# WHATEVER IN IFCONFIG (like eth0 or something
export GLOO_SOCKET_IFNAME=enp0s31f6

world_size=1
batch_size=192
dataset=cifar
arch=alexnet
rank=0

python cifar_resnet.py \
    --rank $rank \
    --world_size $world_size \
    --ip 10.2.1.251 \
    --port 22000 \
    --batch_size $batch_size \
    --epochs 10 \
    --arch $arch \
    --log exp/log_${dataset}_${arch}_world${world_size}_rank${rank}_batch${batch_size}.txt
