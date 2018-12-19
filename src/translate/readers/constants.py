from enum import Enum, unique


@unique
class ReaderType(Enum):
    TRAIN = 0
    TEST = 1
    DEV = 2
    VIS = 3


@unique
class ReaderLevel(Enum):
    WORD = 0
    BPE = 1
    CHAR = 2
