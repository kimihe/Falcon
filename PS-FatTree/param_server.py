# -*- coding: utf-8 -*-
import argparse
import math
import os
import random
import sys
import datetime

import torch
import torch.distributed as dist
from torch.multiprocessing import Process as TorchProcess
from torch.utils.data import DataLoader
from torchvision import datasets

from model_alexnet import alexnet
from torchvision import transforms
from model_utils import test_model

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29000')
parser.add_argument('--this-rank', type=int, default=0)
parser.add_argument('--learners', type=str, default='1-2')

# 模型与数据集
parser.add_argument('--data-dir', type=str, default='./data')
parser.add_argument('--data-name', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='AlexNet')
parser.add_argument('--save-path', type=str, default='./')

# 参数信息
parser.add_argument('--epochs', type=int, default=20)

args = parser.parse_args()


# noinspection PyTypeChecker
def run(rank, model, train_pics, train_bsz):
    workers = [int(v) for v in str(args.learners).split('-')]
    _group = [w for w in workers].append(rank)
    group = dist.new_group(_group)

    for p in model.parameters():
        scatter_p_list = [p.data for _ in range(len(workers) + 1)]
        dist.scatter(tensor=p.data, scatter_list=scatter_p_list, group=group)

    print('Model Sent Finished!')

    print('Begin!')

    transform = transforms.Compose(
        [transforms.Resize(128),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    tmp = [(0, 0) for _ in range(int(math.ceil(train_pics / (len(workers) * train_bsz))))]

    pre_time = datetime.datetime.now()
    for epoch in range(args.epochs):
        for batch_idx, (_, _) in enumerate(tmp):
            for param in model.parameters():
                tensor = torch.zeros_like(param.data)

                # FIXME FIXED：gather_list中的每个Tensor都必须是新的对象，否则会出问题
                gather_list = [torch.zeros_like(param.data) for _ in range(len(workers) + 1)]
                dist.gather(tensor=tensor, gather_list=gather_list, group=group)
                tensor = sum(gather_list) / len(workers)
                param.data -= tensor
                scatter_list = [param.data for _ in range(len(workers) + 1)]
                dist.scatter(tensor=tensor, scatter_list=scatter_list, group=group)


            print('Done {}/{}!'.format(batch_idx, len(tmp)))
        print('Done Epoch {}/{}!'.format(epoch + 1, args.epochs))

    end_time = datetime.datetime.now()
    # 测试ps的模型准确率
    h, remainder = divmod((end_time-pre_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)

    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=False,
                                    transform=transform)
    criterion = torch.nn.CrossEntropyLoss()
    test_data = DataLoader(test_dataset, batch_size=128, shuffle=True)

    test_loss, acc = test_model(dist.get_rank(), model, test_data, criterion=criterion)
    print('total time ' + str(time_str))
    f = open('./result_' + str(rank) + '_' + args.model + '.txt', 'a')
    f.write('Rank: ' + str(rank) +
            ', \tEpoch: ' + str(args.epochs) +
            ', \tTestLoss: ' + str(test_loss) +
            ', \tTestAcc: ' + str(acc) +
            ', \tTotalTime: ' + str(time_str) + '\n')
    f.close()


def init_processes(rank, size,
                   model, train_pics, train_bsz,
                   fn, backend='tcp'):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, model, train_pics, train_bsz)


if __name__ == '__main__':
    # 随机数设置
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    workers = [int(v) for v in str(args.learners).split('-')]

    model = alexnet(num_classes=10)
    train_pics = 50000
    train_bsz = 64

    train_bsz /= len(workers)
    train_bsz = int(train_bsz)

    world_size = len(str(args.learners).split('-')) + 1
    this_rank = args.this_rank

    p = TorchProcess(target=init_processes, args=(this_rank, world_size,
                                                  model, train_pics, train_bsz,
                                                  run))
    p.start()
    p.join()
