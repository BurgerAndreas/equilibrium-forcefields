import torch
import numpy as np


def reorder_dataset(dataset, batch_size):
    """
    order should be s.t. consecutive data steps are consecutive in the dataset
    even when using batches.
    e.g. for batch size 2, and len(dataset) = 10
    order = [0, x, 1, x, 2, x, 3, x, 4, x]
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    """
    steps_per_batch = len(dataset) // batch_size
    order = []
    for i in range(steps_per_batch):
        for j in range(batch_size):
            order.append(j * steps_per_batch + i)
    # append the remaining elements in the dataset
    for i in range(steps_per_batch * batch_size, len(dataset)):
        order.append(i)

    assert len(order) == len(dataset), f"{len(order)} != {len(dataset)}"

    # reorder the dataset
    dataset = [dataset[i] for i in order]
    # dataset_idx = [dataset[i].idx for i in range(len(dataset))]
    return dataset
