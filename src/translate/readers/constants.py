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

    @staticmethod
    def get_granularity(name):
        """
        Given the :param name: of one of the ReaderLevel enums this function returns the equivalent enum object.
        :return:
        """
        if name.lower() == "word":
            return ReaderLevel.WORD
        elif name.lower() == "char":
            return ReaderLevel.CHAR
        elif name.lower() == "bpe":
            return ReaderLevel.BPE
        else:
            raise ValueError("The requested value {} does not exist".format(name))

@unique
class InstancePartType(Enum):
    """
    Defines different types of objects that can be emitted from the DataReader.next() calls
    """
    ListId = 1
    Tensor = 2
    TransformerSrcMask = 3
    TransformerTgtMask = 4


@unique
class LanguageIdentifier(Enum):
    """
    Defines different supported languages specifically for pre-processing step.
    """
    en = 0
    de = 1
    es = 2
    pt = 3
    fr = 4
    it = 5
    nl = 6
