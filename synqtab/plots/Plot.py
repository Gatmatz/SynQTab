from abc import ABC, abstractmethod

class Plot(ABC):
    def __init__(self, params:dict):
        self.params = params

    @abstractmethod
    def read_data(self):
        raise NotImplementedError("Subclasses must implement read_data method")
    
    @abstractmethod
    def run(self):
        raise NotImplementedError("Subclasses must implement run method")