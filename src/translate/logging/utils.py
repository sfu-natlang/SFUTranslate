"""
Provides a uniquely configured logger to the project files
"""
import logging

__author__ = "Hassan S. Shavarani"

logger = logging.getLogger('NatlangMT')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False
