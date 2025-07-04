#!/usr/bin/env python

import argparse
import os
import importlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list")
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument(
        'dataset_name',
        help='The name of the dataset to download')
    list_caches_parser = subparsers.add_parser("list-caches")
    delete_cache_parser = subparsers.add_parser("delete-cache")
    delete_cache_parser.add_argument(
        'cache_name',
        help='The name of the cache to delete')

    args = parser.parse_args()

    if args.command == "list":
        datasets_dir = "ann_benchmarks/datasets"
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
        for f in os.listdir(datasets_dir):
            if f.endswith(".py") and f != "__init__.py" and f != "base.py":
                print(f.replace(".py", ""))
    elif args.command == "download":
        dataset_module = importlib.import_module(f"ann_benchmarks.datasets.{args.dataset_name}")
        dataset = dataset_module.instantiate_dataset()
        dataset.download()
    elif args.command == "list-caches":
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for f in os.listdir(cache_dir):
            if f.endswith(".ann"):
                print(f.replace(".ann", ""))
    elif args.command == "delete-cache":
        cache_name = args.cache_name
        cache_path = f"cache/{cache_name}.ann"
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Deleted cache: {cache_name}")
        else:
            print(f"Cache not found: {cache_name}")
