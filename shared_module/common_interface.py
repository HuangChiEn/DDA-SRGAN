from abc import ABC, abstractmethod

# common methods which should be overwrite by derived SR class 
class SR_base_model(ABC):
    
    @abstractmethod
    def training():
        return NotImplemented
        
    @abstractmethod
    def generating():
        return NotImplemented
    
    @abstractmethod
    def evaluation():
        return NotImplemented
    