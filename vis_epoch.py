
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def format_log(file_name, cifar=False):

    with open(file_name) as f:
        lines = f.readlines()
        print("Line: {}".format(len(lines)))

        res = {"Accuracy": []}
        if cifar:
            p = re.compile(r'.+\* Prec.+')
            for line in lines:
                s = re.search(p, line)
                if s:
                    res['Accuracy'].append(float(line.split()[-1]))
        else:
            p = re.compile(r'^Test set: .+')
            for line in lines:
                s = re.search(p, line)
                if s:
                    items = re.split(r'[,(% ]', line)
                    res['Accuracy'].append(float(items[9]))
    
    return res


def format_logs(file_names, cifar=False):
    ress = []
    for file_name in file_names:
        ress.append(format_log(file_name, cifar))
    return ress


def vis(save_name_pre, logs, labels, cifar):

    ress = format_logs(logs, cifar)
    for item_key in ress[0].keys():
        cur_title = "{}_{}".format(save_name_pre, item_key)
        print("Current: {}".format(item_key))
        
        plt.clf()
        plt.title(cur_title)
        plt.xlabel("Epoch")
        plt.ylabel(item_key)
        
        for idx, log_data in enumerate(ress):
            valid = log_data[item_key]
            plt.plot(range(1, len(valid) + 1), valid, label = labels[idx], marker='.')
        
        plt.legend()
        plt.savefig("log/{}.png".format(cur_title))


if __name__ == "__main__":

    logs = [
            "log/log_mnist_world1_rank0_batch384.txt",
            "log/log_mnist_world2_rank0_batch384.txt",
            "log/log_mnist_world3_rank0_batch384.txt"
    ]
    labels = [
            "world1",
            "world2",
            "world3"
    ]

    save_name_pre = "mnist_epoch_acc_batch384"
    vis(save_name_pre, logs, labels, cifar=False)


