import torch
from torch.utils.data import DataLoader


class DataLoaderIter:
    def __init__(self, loader, auto_reset=True):
        self.loader = loader
        self.iter = iter(self.loader)
        self.auto_reset = auto_reset

    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            if self.auto_reset:
                self.reset()
                return next(self.iter)
            else:
                return

    def reset(self):
        self.iter = iter(self.loader)

    def __len__(self):
        return len(self.loader)


class CUDAPreFetchIter:
    def __init__(self, loader, auto_reset=True):
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.preload()
        self.auto_reset = auto_reset

    def preload(self):
        try:
            one_batch = next(self.iter)
        except StopIteration:
            if self.auto_reset:
                self.reset()
                one_batch = next(self.iter)
            else:
                return
        with torch.cuda.stream(self.stream):
            if isinstance(one_batch, torch.Tensor):
                self.next_batch = one_batch.cuda(non_blocking=True)
            else:
                next_batch = []
                for item in one_batch:
                    if isinstance(item, torch.Tensor):
                        next_batch.append(item.cuda(non_blocking=True))
                    elif isinstance(item, list) or isinstance(item, tuple):
                        next_batch.append([t.cuda(non_blocking=True) for t in item])
                self.next_batch = next_batch

    def reset(self):
        self.iter = iter(self.loader)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        current_batch = self.next_batch
        self.preload()
        return current_batch

    def __len__(self):
        return len(self.loader)


def get_data_iterator(loader, cuda_prefetch=False, verbose=False):
    assert type(loader) is DataLoader
    if cuda_prefetch:
        if verbose:
            print('use CUDA-prefetch data iterator!')
        return CUDAPreFetchIter(loader)
    else:
        return DataLoaderIter(loader)


if __name__ == '__main__':
    data_iter = get_data_iterator(DataLoader(...))
    img, label = data_iter.next()

    while img is not None:
        # processing...
        img, label = data_iter.next()

