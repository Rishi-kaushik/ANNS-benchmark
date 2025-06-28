# ANN Benchmarks

This project benchmarks popular Approximate Nearest Neighbor (ANN) search algorithms on common datasets.

## Getting Started

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Manage datasets:**

    Use the `dataset_manager.py` CLI tool to download and manage datasets.

    *   **List available datasets:**
        ```bash
        python dataset_manager.py list
        ```

    *   **Download a dataset (e.g., SIFT):**
        ```bash
        python dataset_manager.py download sift
        ```

3.  **Run benchmarks:**

    ```bash
    python -m ann_benchmarks.main --algorithm hnsw --dataset sift
    ```
    To force a rebuild of the index even if a cached version exists, use the `--rebuild` flag:
    ```bash
    python -m ann_benchmarks.main --algorithm hnsw --dataset sift --rebuild
    ```

4.  **Manage cached indexes:**

    The benchmark runner now caches built indexes to speed up subsequent runs.

    *   **List available caches:**
        ```bash
        python dataset_manager.py list-caches
        ```

    *   **Delete a specific cache (e.g., for SIFT and HNSW):**
        ```bash
        python dataset_manager.py delete-cache sift-hnsw
        ```

## Adding New Algorithms and Datasets

*   **Algorithms:** To add a new algorithm, create a new file in `ann_benchmarks/algorithms` that inherits from `BaseAlgorithm` and implements the `fit`, `query`, `save`, and `load` methods.
*   **Datasets:** To add a new dataset, create a new file in `ann_benchmarks/datasets` that inherits from `BaseDataset` and implements the `get_train_data`, `get_test_data`, `get_ground_truth`, `get_dimensions`, `get_distance_metric`, and `download` methods.