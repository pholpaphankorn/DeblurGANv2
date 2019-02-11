import os
from functools import partial
from glob import glob
from hashlib import sha1
from typing import Callable, Iterable, Optional, Tuple

import cv2
import numpy as np
from glog import logger
from joblib import Parallel, cpu_count, delayed
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm

from aug import get_transforms


def subsample(data: Iterable, bounds: Tuple[float, float], hash_fn: Callable, n_buckets=100, salt=''):
    data = list(data)
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)

    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    msg = f'Subsampling buckets from {lower_bound} to {upper_bound}, total buckets number is {n_buckets}'
    if salt:
        msg += f'; salt is {salt}'
    logger.info(msg)
    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])


def hash_from_paths(x: Tuple[str, str], salt: str = '') -> str:
    path_a, path_b = x
    names = ''.join(map(os.path.basename, (path_a, path_b)))
    return sha1(f'{names}_{salt}'.encode()).hexdigest()


def split_into_buckets(data: Iterable, n_buckets: int, hash_fn: Callable, salt=''):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])


class PairedDataset(Dataset):
    def __init__(self,
                 files_a: Tuple[str],
                 files_b: Tuple[str],
                 transform: Callable,
                 preload: bool = True,
                 preload_size: Optional[int] = 0):

        assert len(files_a) == len(files_b)

        self.preload = False
        self.data_a = files_a
        self.data_b = files_b
        self.transform = transform

        if preload:
            preload_fn = partial(self._bulk_preload, preload_size=preload_size)
            self.data_a, self.data_b = map(preload_fn, (self.data_a, self.data_b))
            self.preload = True

    def _bulk_preload(self, data: Iterable[str], preload_size: int):
        jobs = [delayed(self._preload)(x, preload_size=preload_size) for x in data]
        return Parallel(n_jobs=cpu_count(), backend='threading')(tqdm(jobs, desc='preloading images'))

    @staticmethod
    def _read_img(x: str):
        img = cv2.imread(x)
        if img is None:
            logger.warning(f'Can not read image {x} with OpenCV, switching to scikit-image')
            img = imread(x)
        return img

    def _preload(self, x: str, preload_size: int):
        img = self._read_img(x)
        h, w, *_ = img.shape
        h_scale = preload_size / h
        w_scale = preload_size / w
        scale = max(h_scale, w_scale)
        img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        assert min(img.shape[:2]) >= preload_size, f'weird img shape: {img.shape}'
        return img

    def _preprocess(self, img):
        img = img.astype('float32') / 255
        img = np.transpose(img, (2, 0, 1))
        return img

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        a, b = self.data_a[idx], self.data_b[idx]
        if not self.preload:
            a, b = map(self._read_img, (a, b))
        a, b = map(self._preprocess, self.transform(a, b))
        return {'a': a, 'b': b}

    @staticmethod
    def from_config(config):
        files_a, files_b = map(lambda x: sorted(glob(config[x], recursive=True)), ('files_a', 'files_b'))
        transform = get_transforms(size=config['size'], scope=config['scope'], crop=config['crop'])
        # ToDo: make augmentations more customizible via transform

        hash_fn = hash_from_paths
        # ToDo: add more hash functions

        data = subsample(data=zip(files_a, files_b),
                         bounds=config.get('bounds', (0, 1)),
                         hash_fn=hash_fn)

        files_a, files_b = map(list, zip(*data))

        return PairedDataset(files_a=files_a,
                             files_b=files_b,
                             preload=config['preload'],
                             preload_size=config['preload_size'],
                             transform=transform)
