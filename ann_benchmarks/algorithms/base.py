
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def __init__(self, metric, dimensions):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def query(self, q, n):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass
