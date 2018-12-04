import argparse
import os
import sys
import time

from torch.multiprocessing import Process as TorchProcess
import torch.distributed as dist

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--ps-ip', type=str, default='127.0.0.1')
parser.add_argument('--ps-port', type=str, default='29000')
parser.add_argument('--this-rank', type=int, default=0)
parser.add_argument('--learners', type=str, default='1-2-3-4')

args = parser.parse_args()

'''
def run(rank, workers):
    pass

'''


def init_processes(rank, size, backend='tcp'):
    os.environ['MASTER_ADDR'] = args.ps_ip
    os.environ['MASTER_PORT'] = args.ps_port
    dist.init_process_group(backend, rank=rank, world_size=size)
    # fn(rank, workers)


if __name__ == '__main__':
    workers = [int(v) for v in str(args.learners).split('-')]
    world_size = len(workers) + 1

    this_rank = args.this_rank

    p = TorchProcess(target=init_processes, args=(this_rank, world_size))
    p.start()
    p.join()