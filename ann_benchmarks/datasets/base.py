
from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def get_train_data(self):
        pass

    @abstractmethod
    def get_test_data(self):
        pass

    @abstractmethod
    def get_ground_truth(self):
        pass

    @abstractmethod
    def get_dimensions(self):
        pass

    @abstractmethod
    def get_distance_metric(self):
        pass
