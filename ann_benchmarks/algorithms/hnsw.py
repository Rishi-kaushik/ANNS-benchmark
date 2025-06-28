
import faiss
from ann_benchmarks.algorithms.base import BaseAlgorithm

class HNSW(BaseAlgorithm):
    def __init__(self, metric, dimensions):
        self.index = faiss.IndexHNSWFlat(dimensions, 32)

    def fit(self, X):
        self.index.add(X)

    def query(self, q, n):
        distances, labels = self.index.search(q.reshape(1, -1), n)
        # faiss returns a 2D array for labels, even for a single query.
        # The first dimension corresponds to the query, so we return labels[0].
        return labels[0]

    def save(self, filename):
        faiss.write_index(self.index, filename)

    def load(self, filename):
        self.index = faiss.read_index(filename)

def instantiate_algorithm(metric, dimensions):
    return HNSW(metric, dimensions)
