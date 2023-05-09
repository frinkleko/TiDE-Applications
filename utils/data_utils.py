import logging

import torch

from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'Sample':
        trainset = None
        testset = None

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(
        trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader