#/bin/bash

python train_dist.py \
    --rank 0 \
    --world-size 2 \
    --ip 219.224.168.78 \
    --port 22000 \
    #--batch_size 64 \
    #--epochs 10 
