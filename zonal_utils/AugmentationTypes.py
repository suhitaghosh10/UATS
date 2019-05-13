from enum import Enum

class AugmentTypes(Enum):
    ROTATE =  0
    FLIP_HORIZ = 1
    TRANSLATION_3D = 2
    SCALE = 3
    ORIGINAL = 4