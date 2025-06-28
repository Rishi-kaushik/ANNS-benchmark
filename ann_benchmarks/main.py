
import argparse
from ann_benchmarks.runner import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm',
        help='The algorithm to run',
        required=True)
    parser.add_argument(
        '--dataset',
        help='The dataset to run on',
        required=True)
    args = parser.parse_args()
    run_benchmark(args.dataset, args.algorithm)
