
# ANN Benchmarks

This project benchmarks popular Approximate Nearest Neighbor (ANN) search algorithms on common datasets.

## Getting Started

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download datasets:**

    You need to download the datasets you want to use. For example, to download the SIFT dataset:

    ```bash
    wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    tar -xzf sift.tar.gz
    # You will need to convert the data to HDF5 format.
    python create_sift_hdf5.py sift
    ```

3.  **Run benchmarks:**

    ```bash
    python -m ann_benchmarks.main --algorithm hnsw --dataset sift
    ```

## Adding New Algorithms and Datasets

*   **Algorithms:** To add a new algorithm, create a new file in `ann_benchmarks/algorithms` that inherits from `BaseAlgorithm` and implements the required methods.
*   **Datasets:** To add a new dataset, create a new file in `ann_benchmarks/datasets` that inherits from `BaseDataset` and implements the required methods.
