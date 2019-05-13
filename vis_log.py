
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def format_log(file_name):

    with open(file_name) as f:
        lines = f.readlines()
        print("Line: {}".format(len(lines)))

        p1 = re.compile(r'.+epoch \d+ iter .+')
        p2 = re.compile(r'^Test set: .+')
        res = {
            "train_loss": [], 
            "train_cost": [], 
            "test_loss": [],
            "test_acc": []
        }

        for line in lines:
            s1 = re.search(p1, line)
            s2 = re.search(p2, line)
            if s1:
                items = line.split()
                res['train_loss'].append(float(items[8]))
                res['train_cost'].append(float(items[10][:-1]))
            if s2:
                items = re.split(r'[,(% ]', line)
                res['test_loss'].append(float(items[4]))
                res['test_acc'].append(float(items[9]))
    
    return res


def format_logs(file_names):
    ress = []
    for file_name in file_names:
        ress.append(format_log(file_name))
    return ress


def vis(save_name_pre, logs, labels):

    ress = format_logs(logs)
    for item_key in ress[0].keys():
        cur_title = "{}_{}".format(save_name_pre, item_key)
        print("Current: {}".format(item_key))
        
        plt.clf()
        plt.title(cur_title)
        plt.xlabel("epoch")
        plt.ylabel(item_key)
        
        for idx, log_data in enumerate(ress):
            valid = log_data[item_key]
            plt.plot(range(1, len(valid) + 1), valid, label = labels[idx])
        
        plt.legend()
        plt.savefig("log/{}.png".format(cur_title))


if __name__ == "__main__":

    logs = [
            "log/log_mnist_node1_batch32.txt",
            "log/log_mnist_node1_batch64.txt",
            "log/log_mnist_node1_batch128.txt"
    ]
    labels = [
            "batch32",
            "batch64",
            "batch128"
    ]

    save_name_pre = "test"
    vis(save_name_pre, logs, labels)


