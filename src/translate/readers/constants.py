"""
Contains constant Enum values required for the different dataset readers in the project
"""

from enum import Enum, unique

__author__ = "Hassan S. Shavarani"


@unique
class ReaderType(Enum):
    """
    Defines the dataset type (will be used in case dataset wants to create the vocabulary objects)
    """
    TRAIN = 0
    TEST = 1
    DEV = 2


@unique
class ReaderLevel(Enum):
    """
    Defines different granularity levels in reading the dataset files
    """
    WORD = 0
    BPE = 1
    CHAR = 2
