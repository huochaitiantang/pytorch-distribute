#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms

import time
import argparse
import logging


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net_old(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def partition_dataset(batch_size):
    """ Partitioning MNIST """
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = batch_size / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=int(bsz), shuffle=True)
    return train_set, bsz


def dis_train_dataset(batch_size):
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    bsz = int(batch_size / dist.get_world_size())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = bsz, 
        shuffle = (train_sampler is None),
        sampler = train_sampler)
     
    return train_loader, bsz


def init_logger(log_file):
    
    logger = logging.getLogger("DIS")
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(filename)s:%(lineno)d" \
                                  "-%(levelname)s-%(message)s")

    # log file stream
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # log console stream
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger



def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def format_second(secs):
    return "{:0>2}:{:0>2}:{:0>2}(h:m:s)".format( \
            int(secs / 3600), int((secs % 3600) / 60), int(secs % 60))


def get_test_dataset():
    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False)
    return test_loader


def test(model, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return


def run(rank, size, batch_size, epochs, logger):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset(batch_size)
    test_loader = get_test_dataset()
    #train_set, bsz = dis_train_dataset(64)
    model = Net()
    model.train()
    
    logger.info("Model: {}".format(list(model.parameters())[0].mean()))

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    tt = 0.
    all_conn_tt = 0.
    for epoch in range(epochs):
        t1 = time.time()
        epoch_loss = 0.0
        epoch_conn_tt = 0.
        for batch_idx, (data, target) in enumerate(train_set):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)

            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            
            conn_t1 = time.time()
            average_gradients(model)
            epoch_conn_tt += time.time() - conn_t1
            
            optimizer.step()

            if batch_idx % 10 == 0: 
                logger.info('Rank {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    dist.get_rank(), epoch, batch_idx * len(data), len(train_set.dataset),
                    100. * batch_idx / len(train_set), loss.item()))

        t2 = time.time()
        tt += t2 - t1
        all_conn_tt += epoch_conn_tt

        logger.info('Rank {} epoch {} iter {} loss {:.5f} cost {:.5f}s ' \
              '(conn: {:.5f} s, ave:{:.5f}s/time)'.format( \
              dist.get_rank(), epoch, len(train_set), epoch_loss / num_batches, \
              t2 - t1, epoch_conn_tt, epoch_conn_tt/len(train_set)))
        
        test(model, test_loader, logger)
    
    logger.info("Train {} epochs, cost {:.5f} s({} conn: {:.5f}s " \
          "ave{:.5f}s/epoch), average{:.5f} s / epoch ".format( \
           epochs, tt, format_second(tt), all_conn_tt, \
           all_conn_tt /epochs, tt / epochs))


def init_process(args, logger):
    #Initialize the distributed environment.
    dist.init_process_group(
        backend = args.backend,
        init_method = "tcp://{}:{}".format(args.ip, args.port),
        rank = args.rank,
        world_size = args.world_size)
    #Run
    run(args.rank, args.world_size, args.batch_size, args.epochs, logger)


if __name__ == "__main__":
    '''
    size = 1
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--backend', type=str, default="gloo")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--log', type=str, default='log.txt')
    
    args = parser.parse_args()
    logger = init_logger(args.log)
    logger.info("Agrs: {}".format(args))
    
    init_process(args, logger)
