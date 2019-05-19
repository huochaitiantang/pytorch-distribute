
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def format_log(file_name, cifar=False):

    with open(file_name) as f:
        lines = f.readlines()
        if cifar:
            acc = float(re.split(r'[()]', lines[-4])[-2])
            all_t = float(re.split(r'[ s]', lines[-2])[8])
            conn_t = float(re.split(r'[ s]', lines[-1])[8])
            train_t = all_t - conn_t
        else:
            acc = float(re.split(r'[(%]', lines[-3])[-2])
            all_t = float(re.split(r'[ s]', lines[-1])[8])
            conn_t = float(re.split(r'[ s]', lines[-1])[13])
            train_t = all_t - conn_t
        print(conn_t, all_t, conn_t/all_t)

    return acc, all_t, conn_t, train_t, conn_t / all_t


def format_logss(file_namess, cifar=False):
    ress = {
        "Accuracy": [],
        "Cost": [],
        "Cost_Conn": [],
        "Cost_Train": [],
        "Conn_Ratio": []
    }
    for file_names in file_namess:
        batch_acc = []
        batch_all_t = []
        batch_conn_t = []
        batch_train_t = []
        batch_conn_ratio = []
        for file_name in file_names:
            acc, all_t, conn_t, train_t, conn_ratio = format_log(file_name, cifar)
            batch_acc.append(acc)
            batch_all_t.append(all_t)
            batch_conn_t.append(conn_t)
            batch_train_t.append(train_t)
            batch_conn_ratio.append(conn_ratio)

        ress['Accuracy'].append(batch_acc)
        ress['Cost'].append(batch_all_t)
        ress['Cost_Conn'].append(batch_conn_t)
        ress['Cost_Train'].append(batch_train_t)
        ress['Conn_Ratio'].append(batch_conn_ratio)
    return ress


def vis(save_name_pre, ress, labels, xs, valid_keys):

    print(ress)
    
    for item_key in ress.keys():
        if item_key not in valid_keys:
            continue

        cur_title = "{}_{}".format(save_name_pre, item_key)
        print("Current: {}".format(item_key))

        assert(len(ress[item_key]) == len(labels))
        assert(len(ress[item_key][0]) == len(xs))

        plt.clf()
        plt.title(cur_title)
        plt.xlabel("Batch")
        if item_key == "Cost" or item_key == "Cost_Train":
            plt.ylabel(item_key + "/s")
        else:
            plt.ylabel(item_key)
        
        for idx, batch_data in enumerate(ress[item_key]):
            plt.plot(xs, batch_data, label = labels[idx], marker='.')
        
        plt.legend()
        plt.savefig("log/pic/{}.png".format(cur_title))


def get_mnist_logs(ignore_rank = False):
    batches = [96, 192, 384, 768, 1536]
    worlds = [1, 2, 3]
    logs = []
    for world in worlds:
        rank_cnt = 1 if ignore_rank else world
        for rank in range(rank_cnt):
            logs.append(["log/mnist/log_mnist_world{}_rank{}_batch{}.txt"\
                         .format(world, rank, batch) for batch in batches])
    return batches, logs


def get_cifar_logs(ignore_rank = False):
    batches = [96, 192, 384]
    worlds = [1, 2, 3]
    logs = []
    for world in worlds:
        rank_cnt = 1 if ignore_rank else world
        for rank in range(rank_cnt):
            logs.append(["log/cifar/log_cifar_alexnet_world{}_rank{}_batch{}.txt"\
                         .format(world, rank, batch) for batch in batches])
    return batches, logs
             

def vis_mnist_acc():
    batches, logs = get_mnist_logs(ignore_rank = True)
    labels = ["world1", "world2", "world3"]
    ress = format_logss(logs, cifar=False)
    vis("mnist_batch", ress, labels, batches, ["Accuracy"])
    

def vis_cifar_acc():
    batches, logs = get_cifar_logs(ignore_rank = True)
    labels = ["world1", "world2", "world3"]
    ress = format_logss(logs, cifar=True)
    vis("cifar_batch", ress, labels, batches, ["Accuracy"])
    

def vis_mnist_cost():
    batches, logs = get_mnist_logs()
    labels = ["world1_rank0", "world2_rank0", "world2_rank1",
              "world3_rank0", "world3_rank1", "world3_rank2"]
    ress = format_logss(logs, cifar=False)
    vis("mnist_batch", ress, labels, batches, ["Cost", "Cost_Train", "Conn_Ratio"])


def vis_cifar_cost():
    batches, logs = get_cifar_logs()
    labels = ["world1_rank0", "world2_rank0", "world2_rank1",
              "world3_rank0", "world3_rank1", "world3_rank2"]
    ress = format_logss(logs, cifar=True)
    vis("cifar_batch", ress, labels, batches, ["Cost", "Cost_Train", "Conn_Ratio"])


if __name__ == "__main__":

    vis_mnist_acc()
    vis_mnist_cost()
    vis_cifar_acc()
    vis_cifar_cost()


