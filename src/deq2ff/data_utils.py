import torch
import numpy as np


def reorder_dataset(dataset, batch_size):
    """
    order should be s.t. consecutive test steps have consecutive idxs
    even when using batches.
    e.g. for batch size 2, and len(dataset) = 10
    order = [0, ., 1, ., 2, ., 3, ., 4, .]
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

    # # patches
    # test_patch_size_select = None
    # test_patch_size = len(dataset)
    # test_patches = 2

    # if test_patch_size_select is None:
    #     test_patch_size_select = 1000
    # # assert test_patch_size_select <= test_patch_size, \
    # #     f"Warning: test_patch_size_select ({test_patch_size_select}) is greater than test_patch_size ({test_patch_size})."
    # start_idx = np.linspace(0, test_patch_size - test_patch_size_select, test_patches, dtype=int)
    # test_indices = np.hstack([np.arange(s, s + test_patch_size_select) for s in start_idx])

    # order = np.asarray(order)[test_indices]

    # reorder the dataset
    dataset = [dataset[i] for i in order]
    # dataset_idx = [dataset[i].idx for i in range(len(dataset))]
    return dataset
