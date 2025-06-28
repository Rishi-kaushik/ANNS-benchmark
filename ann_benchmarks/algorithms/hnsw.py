
import faiss
from ann_benchmarks.algorithms.base import BaseAlgorithm

class HNSW(BaseAlgorithm):
    def __init__(self, metric, dimensions):
        self.index = faiss.IndexHNSWFlat(dimensions, 32)

    def fit(self, X):
        self.index.add(X)

    def query(self, q, n):
        distances, labels = self.index.search(q.reshape(1, -1), n)
        return labels[0]

def instantiate_algorithm(metric, dimensions):
    return HNSW(metric, dimensions)
