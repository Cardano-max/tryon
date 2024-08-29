from abc import ABC, abstractmethod

class BaseMaskAdapter(ABC):
    @abstractmethod
    def generate_mask(self, img, category):
        pass