
import numpy as np
import h5py
import argparse
import os

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sift_dir', help='Directory containing the SIFT dataset files')
    args = parser.parse_args()

    sift_base_path = os.path.join(args.sift_dir, 'sift_base.fvecs')
    sift_query_path = os.path.join(args.sift_dir, 'sift_query.fvecs')
    sift_groundtruth_path = os.path.join(args.sift_dir, 'sift_groundtruth.ivecs')

    train_data = fvecs_read(sift_base_path)
    test_data = fvecs_read(sift_query_path)
    ground_truth = ivecs_read(sift_groundtruth_path)

    with h5py.File('sift-128-euclidean.hdf5', 'w') as f:
        f.create_dataset('train', data=train_data)
        f.create_dataset('test', data=test_data)
        f.create_dataset('neighbors', data=ground_truth)

    print("Successfully created sift-128-euclidean.hdf5")
