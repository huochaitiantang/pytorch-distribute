# pytorch-distribute
Pytorch distributed training for CPU

## Setting

#### python3 virtual environment
```
sudo apt install virtualenv
virtualenv ~/dispytorch -p python3
source ~/dispytorch/bin/activate
```

#### pytorch 1.1.0 (cpu)
```
pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp35-cp35m-linux_x86_64.whl
pip install torchvision
```
* if python is python3.6, please install as follow
```
pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip install torchvision
```

#### matplotlib
```
pip install matplotlib
```

#### Train

Note: The variable `GLOO_SOCKET_IFNAME` in train_mnist.sh should be eth0 or something output by `ifconfig`

* mnist
```
sh train_mnist.sh
```

* cifar
```
sh train_cifar.sh
```

* multi-cpu(eg, mnist with 2 nodes and node1 ip is 219.224.168.78)
  * node1 start `sh train_mnist.sh` with setting:
  ```
  world_size=2
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
  ```
  
  * node2 start `sh train_mnist.sh` with setting:
  ```
  world_size=2
  batch_size=96
  dataset=mnist
  rank=1
  
  python train_dist.py \
    --rank $rank \
    --world_size $world_size \
    --ip 219.224.168.78 \
    --port 22000 \
    --batch_size $batch_size \
    --epochs 10 \
    --log log/log_${dataset}_world${world_size}_rank${rank}_batch${batch_size}.txt
  ```
