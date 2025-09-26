from typing import Dict, ClassVar
from dataclasses import dataclass
from automated_underwater_area_estimation.segmentation.class_mapping import (
    ClassMappingBase,
)


@dataclass
class EPFLClassMapping(ClassMappingBase):
    """Class mapping for the EPFL Coralscapes dataset with 39 classes."""

    CLASS_NAMES: ClassVar[Dict[int, str]] = {
        1: "seagrass",
        2: "trash",
        3: "other coral dead",
        4: "other coral bleached",
        5: "sand",
        6: "other coral alive",
        7: "human",
        8: "transect tools",
        9: "fish",
        10: "algae covered substrate",
        11: "other animal",
        12: "unknown hard substrate",
        13: "background",
        14: "dark",
        15: "transect line",
        16: "massive/meandering bleached",
        17: "massive/meandering alive",
        18: "rubble",
        19: "branching bleached",
        20: "branching dead",
        21: "millepora",
        22: "branching alive",
        23: "massive/meandering dead",
        24: "clam",
        25: "acropora alive",
        26: "sea cucumber",
        27: "turbinaria",
        28: "table acropora alive",
        29: "sponge",
        30: "anemone",
        31: "pocillopora alive",
        32: "table acropora dead",
        33: "meandering bleached",
        34: "stylophora alive",
        35: "sea urchin",
        36: "meandering alive",
        37: "meandering dead",
        38: "crown of thorn",
        39: "dead clam",
    }
    # List of class IDs that represent coral (any kind - alive, dead, or bleached)
    CORAL_CLASS_IDS: ClassVar[list[int]] = [
        3,   # "other coral dead"
        4,   # "other coral bleached"
        6,   # "other coral alive"
        16,  # "massive/meandering bleached"
        17,  # "massive/meandering alive"
        19,  # "branching bleached"
        20,  # "branching dead"
        21,  # "millepora"
        22,  # "branching alive"
        23,  # "massive/meandering dead"
        25,  # "acropora alive"
        27,  # "turbinaria"
        28,  # "table acropora alive"
        31,  # "pocillopora alive"
        32,  # "table acropora dead"
        33,  # "meandering bleached"
        34,  # "stylophora alive"
        36,  # "meandering alive"
        37,  # "meandering dead"
    ]
