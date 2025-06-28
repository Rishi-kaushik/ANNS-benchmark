
import h5py
import numpy as np
import os
import subprocess
from ann_benchmarks.datasets.base import BaseDataset

class SIFT(BaseDataset):
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.ground_truth = None
        self.dimensions = 128
        self.distance_metric = 'L2'

    def get_train_data(self):
        if self.train_data is None:
            with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
                self.train_data = np.array(f['train'])
        return self.train_data

    def get_test_data(self):
        if self.test_data is None:
            with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
                self.test_data = np.array(f['test'])
        return self.test_data

    def get_ground_truth(self):
        if self.ground_truth is None:
            with h5py.File('sift-128-euclidean.hdf5', 'r') as f:
                self.ground_truth = np.array(f['neighbors'])
        return self.ground_truth

    def get_dimensions(self):
        return self.dimensions

    def get_distance_metric(self):
        return self.distance_metric

    def download(self):
        if not os.path.exists('sift-128-euclidean.hdf5'):
            print("Downloading sift-128-euclidean.hdf5...")
            subprocess.run(["wget", "http://ann-benchmarks.com/sift-128-euclidean.hdf5"])
        else:
            print("sift-128-euclidean.hdf5 already exists.")

def instantiate_dataset():
    return SIFT()
