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

#### Run
```
sh train_dis.sh
```
