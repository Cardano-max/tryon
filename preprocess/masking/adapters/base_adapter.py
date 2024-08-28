from abc import ABC, abstractmethod

class BaseMaskAdapter(ABC):
    @abstractmethod
    def generate_mask(self, category, model_parse, keypoint, width, height):
        pass