from enum import Enum


class ExperimentType(Enum):
    NORMAL = 'normal'
    PRIVACY = 'privacy'
    AUGMENTATION = 'augmentation'
    REBALANCING = 'rebalancing'
    
    def shortname(self) -> str:
        """First three letters in capital.

        Returns:
            str: The shortname of the ExperimentType object.
        """
        return self.value.upper()[:3]
