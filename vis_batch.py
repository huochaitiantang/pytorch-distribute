
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
        plt.savefig("pic/{}.png".format(cur_title))


if __name__ == "__main__":

    cifar_logs = [
        [
            "exp/log_cifar_alexnet_world1_rank0_batch96.txt",
            "exp/log_cifar_alexnet_world1_rank0_batch192.txt",
            "exp/log_cifar_alexnet_world1_rank0_batch384.txt"
        ],
        [
            "exp/log_cifar_alexnet_world2_rank0_batch96.txt",
            "exp/log_cifar_alexnet_world2_rank0_batch192.txt",
            "exp/log_cifar_alexnet_world2_rank0_batch384.txt"
        ],
        [
            "exp/log_cifar_alexnet_world2_rank1_batch96.txt",
            "exp/log_cifar_alexnet_world2_rank1_batch192.txt",
            "exp/log_cifar_alexnet_world2_rank1_batch384.txt"
        ],
        [
            "exp/log_cifar_alexnet_world3_rank0_batch96.txt",
            "exp/log_cifar_alexnet_world3_rank0_batch192.txt",
            "exp/log_cifar_alexnet_world3_rank0_batch384.txt"
        ],
        [
            "exp/log_cifar_alexnet_world3_rank1_batch96.txt",
            "exp/log_cifar_alexnet_world3_rank1_batch192.txt",
            "exp/log_cifar_alexnet_world3_rank1_batch384.txt"
        ],
        [
            "exp/log_cifar_alexnet_world3_rank2_batch96.txt",
            "exp/log_cifar_alexnet_world3_rank2_batch192.txt",
            "exp/log_cifar_alexnet_world3_rank2_batch384.txt"
        ]
    ]


    mnist_logs = [
        [
            "log/log_mnist_world1_rank0_batch96.txt",
            "log/log_mnist_world1_rank0_batch192.txt",
            "log/log_mnist_world1_rank0_batch384.txt",
            "log/log_mnist_world1_rank0_batch768.txt",
            "log/log_mnist_world1_rank0_batch1536.txt",
        ],
        [
            "log/log_mnist_world2_rank0_batch96.txt",
            "log/log_mnist_world2_rank0_batch192.txt",
            "log/log_mnist_world2_rank0_batch384.txt",
            "log/log_mnist_world2_rank0_batch768.txt",
            "log/log_mnist_world2_rank0_batch1536.txt",
        ],
        [
            "log/log_mnist_world2_rank1_batch96.txt",
            "log/log_mnist_world2_rank1_batch192.txt",
            "log/log_mnist_world2_rank1_batch384.txt",
            "log/log_mnist_world2_rank1_batch768.txt",
            "log/log_mnist_world2_rank1_batch1536.txt",
        ],
        [
            "log/log_mnist_world3_rank0_batch96.txt",
            "log/log_mnist_world3_rank0_batch192.txt",
            "log/log_mnist_world3_rank0_batch384.txt",
            "log/log_mnist_world3_rank0_batch768.txt",
            "log/log_mnist_world3_rank0_batch1536.txt",
        ],
        [
            "log/log_mnist_world3_rank1_batch96.txt",
            "log/log_mnist_world3_rank1_batch192.txt",
            "log/log_mnist_world3_rank1_batch384.txt",
            "log/log_mnist_world3_rank1_batch768.txt",
            "log/log_mnist_world3_rank1_batch1536.txt",
        ],
        [
            "log/log_mnist_world3_rank2_batch96.txt",
            "log/log_mnist_world3_rank2_batch192.txt",
            "log/log_mnist_world3_rank2_batch384.txt",
            "log/log_mnist_world3_rank2_batch768.txt",
            "log/log_mnist_world3_rank2_batch1536.txt",
        ],
    ]


    labels_all = [
            "world1_rank0",
            "world2_rank0",
            "world2_rank1",
            "world3_rank0",
            "world3_rank1",
            "world3_rank2"
    ]

    labels_world = [
            "world1",
            "world2",
            "world3"
    ]

    #xs = [96, 192, 384]
    xs = [96, 192, 384, 768, 1536]

    save_name_pre = "mnist_batch"
    #valid_keys = ["Accuracy"]
    #valid_keys = ["Cost", "Cost_Train"]
    valid_keys = ["Conn_Ratio"]

    ress = format_logss(mnist_logs, cifar=False)
    vis(save_name_pre, ress, labels_all, xs, valid_keys)


