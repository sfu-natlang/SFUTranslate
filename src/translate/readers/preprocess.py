"""
The main script for reading the configurations from the config file (passed in in YAML/Ansible format), pre-processing
 the files and storing them in the results directory (creates the directory if does not exist).
  You can run this script using the following bash script [The config file "conf.yaml" will be looked up from
   /path/to/SFUTranslate/resources]:
####################################################################
#!/usr/bin/env bash
cd /path/to/SFUTranslate/src && python -m translate.readers.preprocess conf.yaml
####################################################################
The config file must contain the following parts:
 reader:
     preprocess:
        granularity: indicating the requested granularity level of the resulting dataset as either of CHAR/BPE/WORD
        dataset_directory: the path to the directory containing raw data files
        result_directory: the path to the directory to which the results are to be stored (if not existing will be created)
        source_lang: the bi-letter tag indicating the source language ['en'|'fr'|'de'|...]
        target_lang: the bi-letter tag indicating the target language ['en'|'fr'|'de'|...]
        dataset_prefix: the dataset files prefix (e.g. IWSLT dataset files all begin with "IWSLT17.TED")
"""
import sys

from translate.configs.loader import ConfigLoader
from translate.configs.utils import get_resource_file
from translate.readers.constants import ReaderLevel

__author__ = "Hassan S. Shavarani"


class Preprocess:
    def __init__(self, configs: ConfigLoader):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        """
        granularity = configs.get("reader.preprocess.granularity", "WORD")
        dataset_directory = configs.get("reader.preprocess.dataset_directory", must_exist=True)
        result_directory = configs.get("reader.preprocess.result_directory", must_exist=True)
        src_lang = configs.get("reader.preprocess.source_lang", must_exist=True)
        tgt_lang = configs.get("reader.preprocess.target_lang", must_exist=True)
        dataset_prefix = configs.get("reader.preprocess.dataset_prefix", "")

        if granularity.lower() == "char":
            self.granularity = ReaderLevel.CHAR
        elif granularity.lower() == "bpe":
            self.granularity = ReaderLevel.BPE
        else:
            self.granularity == ReaderLevel.WORD

if __name__ == '__main__':
    # The single point which loads the config file passed to the script
    configurations_path = sys.argv[1]
    opts = ConfigLoader(get_resource_file(configurations_path))
    data_preprocess = Preprocess(opts)
