
import importlib
import time

def run_benchmark(dataset_name, algorithm_name):
    print(f"Running benchmark for {algorithm_name} on {dataset_name}")

    dataset_module = importlib.import_module(f"ann_benchmarks.datasets.{dataset_name}")
    dataset = dataset_module.instantiate_dataset()

    algorithm_module = importlib.import_module(f"ann_benchmarks.algorithms.{algorithm_name}")
    algorithm = algorithm_module.instantiate_algorithm(dataset.get_distance_metric(), dataset.get_dimensions())

    start_time = time.time()
    algorithm.fit(dataset.get_train_data())
    build_time = time.time() - start_time
    print(f"Build time: {build_time:.4f}s")

    start_time = time.time()
    recalls = []
    for i, (query, ground_truth) in enumerate(zip(dataset.get_test_data(), dataset.get_ground_truth())):
        results = algorithm.query(query, 10)
        recall = len(set(results) & set(ground_truth)) / len(ground_truth)
        recalls.append(recall)
    query_time = time.time() - start_time
    avg_recall = sum(recalls) / len(recalls)
    qps = len(dataset.get_test_data()) / query_time

    print(f"Average recall: {avg_recall:.4f}")
    print(f"Queries per second: {qps:.4f}")
