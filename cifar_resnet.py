import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import logging
import torch
import torch.distributed as dist
from random import Random

print_freq = 10

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


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def format_second(secs):
    return "{:0>2}:{:0>2}:{:0>2}(h:m:s)".format( \
            int(secs / 3600), int((secs % 3600) / 60), int(secs % 60))


def build_dataset(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    world_size = dist.get_world_size()
    split_batch_size = int(batch_size / float(world_size))

    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(train_dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_loader = torch.utils.data.DataLoader(
        partition, batch_size=split_batch_size, shuffle=True)

    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False)

    return train_loader, val_loader

def train(train_loader, model, criterion, optimizer, epoch, logger):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    epoch_conn_tt = 0.
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        conn_t1 = time.time()
        average_gradients(model)
        epoch_conn_tt += time.time() - conn_t1

        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1.data, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(
                'Rank: [{rank}] Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    rank=dist.get_rank(), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
    return epoch_conn_tt


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output = model(input)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.data, input.size(0))
            top1.update(prec1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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


def main(args):
    # Init logger
    logger = init_logger(args.log)
    logger.info("Args: {}".format(args))

    # Initialize the distributed environment.
    dist.init_process_group(
        backend = args.backend,
        init_method = "tcp://{}:{}".format(args.ip, args.port),
        rank = args.rank,
        world_size = args.world_size)

    # Build model
    model = resnet.__dict__[args.arch]()

    # Build optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Build criterion
    criterion = nn.CrossEntropyLoss()

    # Build lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[25, 38])
            # optimizer, milestones=[100, 150])

    train_loader, val_loader = build_dataset(args.batch_size)

    tt = 0.
    all_conn_tt = 0.

    # Hard set save directory
    save_dir = './ckpts'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_prec1 = -1
    # Train
    for epoch in range(args.epochs):

        t1 = time.time()

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        epoch_conn_tt = train(
            train_loader, model, criterion, optimizer, epoch, logger)
        lr_scheduler.step()

        t2 = time.time()
        tt += t2 - t1
        all_conn_tt += epoch_conn_tt
        logger.info('Rank: [{}] Epoch: [{}] Cost: {:.5f}s '
              '(Conn: {:.5f}s, Ave: {:.5f}s/iter)'.format(
              dist.get_rank(), epoch,
              t2-t1, epoch_conn_tt, epoch_conn_tt/len(train_loader)))

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        if prec1 > best_prec1:
            best_prec1 = prec1
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(save_dir, 'model.th'))

    logger.info(
        "Train {} epochs, "
        "Total time: {:.5f}s({}), Average: {:.5f}s/epoch".format(
        args.epochs, tt, format_second(tt), tt/args.epochs))
    logger.info(
        "Train {} epochs, "
        "Total conn: {:.5f}s({}), Average: {:.5f}s/epoch".format(
        args.epochs, all_conn_tt, format_second(all_conn_tt),
        all_conn_tt/args.epochs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str)
    parser.add_argument('--port', type=str)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--backend', type=str, default="gloo")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log', type=str, default='log.txt')

    parser.add_argument('--arch', default='resnet20')
    parser.add_argument('--learning-rate', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--print-freq', default=50, type=int)

    args = parser.parse_args()

    main(args)
