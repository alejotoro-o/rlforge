from abc import abstractmethod

class BaseAgent:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def end(self):
        pass

    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def rest(self):
        pass