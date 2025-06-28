
import argparse
import importlib
import time
import os
from tqdm.auto import tqdm
import numpy as np

def run_benchmark(dataset_name, algorithm_name, rebuild):
    print(f"Running benchmark for {algorithm_name} on {dataset_name}")

    dataset_module = importlib.import_module(f"ann_benchmarks.datasets.{dataset_name}")
    dataset = dataset_module.instantiate_dataset()

    algorithm_module = importlib.import_module(f"ann_benchmarks.algorithms.{algorithm_name}")
    algorithm = algorithm_module.instantiate_algorithm(dataset.get_distance_metric(), dataset.get_dimensions())

    cache_fn = f"cache/{dataset_name}-{algorithm_name}.ann"
    if os.path.exists(cache_fn) and not rebuild:
        print("Loading index from cache...")
        algorithm.load(cache_fn)
    else:
        print("Building index...")
        start_time = time.time()
        algorithm.fit(dataset.get_train_data())
        build_time = time.time() - start_time
        print(f"Build time: {build_time:.4f}s")
        print("Saving index to cache...")
        algorithm.save(cache_fn)

    start_time = time.time()
    recalls = {1: [], 3: [], 5: []}
    precisions = {1: [], 3: [], 5: []}
    
    for query, ground_truth in tqdm(zip(dataset.get_test_data(), dataset.get_ground_truth()), total=len(dataset.get_test_data())):
        results = algorithm.query(query, 10)
        
        for k in recalls.keys():
            recall_k = len(set(results[:k]) & set(ground_truth[:k])) / k
            recalls[k].append(recall_k)
            
            precision_k = len(set(results[:k]) & set(ground_truth[:k])) / len(results[:k])
            precisions[k].append(precision_k)
            
    query_time = time.time() - start_time
    
    for k in sorted(recalls.keys()):
        avg_recall_k = np.mean(recalls[k])
        avg_precision_k = np.mean(precisions[k])
        print(f"Average recall@{k}: {avg_recall_k:.4f}")
        print(f"Average precision@{k}: {avg_precision_k:.4f}")
        
    qps = len(dataset.get_test_data()) / query_time
    print(f"Queries per second: {qps:.4f}")
