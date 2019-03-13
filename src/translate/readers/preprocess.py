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
        dataset_directory: the path to the directory containing raw data files
        result_directory: the path to the directory to which the results are to be stored (if not existing will be created)
        source_lang: the bi-letter tag indicating the source language ['en'|'fr'|'de'|...]
        target_lang: the bi-letter tag indicating the target language ['en'|'fr'|'de'|...]
        dataset_type: the type of the data placed inside dataset_directory; possible values [REGULAR | IWSLT]
        to_lower: whether the output preprocessed data needs to be lower-cased or not
#####################################################################
Currently supported pre-processing languages: `English`,`German`,`Spanish`,`Portuguese`,`French`,`Italian`, and `Dutch`.
#####################################################################
The code will eventually create six files in the reader.preprocess.result_directory named:
    - train.{source_lang}
    - train.{target_lang}
    - dev.{source_lang}
    - dev.{target_lang}
    - test.{source_lang}
    - test.{target_lang}
"""
import sys
import xml.etree.ElementTree as ET
from typing import Iterator
from tqdm import tqdm

from translate.configs.loader import ConfigLoader
from translate.configs.utils import get_resource_file
from translate.readers.constants import LanguageIdentifier as LId
from translate.readers.tokenizer import SpaCyTokenizer
from translate.logging.utils import logger
from translate.configs.utils import Path

__author__ = "Hassan S. Shavarani"


class Preprocess:
    def __init__(self, configs: ConfigLoader):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        """
        self.dataset_directory = Path(configs.get("reader.preprocess.dataset_directory", must_exist=True))
        self.result_directory = Path(configs.get("reader.preprocess.result_directory", must_exist=True))
        self.to_lower = configs.get("reader.preprocess.to_lower", False)
        if not self.result_directory.exists():
            self.result_directory.mkdir(parents=True)
        self.src_lang = LId[configs.get("reader.preprocess.source_lang", must_exist=True)]
        self.tgt_lang = LId[configs.get("reader.preprocess.target_lang", must_exist=True)]
        """
        If the dataset_type is set to "IWSLT", the execute function expects the "reader.preprocess.dataset_directory"
        config points to a directory containing the original IWSLT dataset format as:
            For each language pair x-y, the in-domain parallel training data is provided through the following files:
                train.tags.x-y.x
                train.tags.x-y.y
            The transcripts are given as pure text (UTF8 encoding), one or more
            sentences per line, and are aligned (at language pair level, not across pairs).
            For tuning/development purposes, the following files are released:
                {IWSLT YEAR TED}.dev{YEAR}.x-y.x.xml
                {IWSLT YEAR TED}.dev{YEAR}.x-y.y.xml
                {IWSLT YEAR TED}.tst201[012345].x-y.x.xml
                {IWSLT YEAR TED}.tst201[012345].x-y.y.xml
        """
        self.dataset_type = configs.get("reader.preprocess.dataset_type", must_exist=True)
        self.tokenizer = SpaCyTokenizer()

    def execute(self):
        """
        The main function which needs to be called to start preprocessing the directory pointed through
         reader.preprocess.dataset_directory in the config file
        :return:
        """
        if self.dataset_type.lower() == "iwslt":
            self._execute_iwslt_prep()
        elif self.dataset_type.lower() == "wmt":
            self._execute_wmt_prep()
        else:
            raise NotImplementedError

    def _execute_iwslt_prep(self):
        src_train_files = self._access_text_file_lines(self.dataset_directory.rglob(
            "train.tags*.%s" % self.src_lang.name))
        tgt_train_files = self._access_text_file_lines(self.dataset_directory.rglob(
            "train.tags*.%s" % self.tgt_lang.name))
        src_dev_files = self._access_xml_seg_tags(sorted(self.dataset_directory.rglob("*.dev*.%s.xml" % self.src_lang.name)))
        tgt_dev_files = self._access_xml_seg_tags(sorted(self.dataset_directory.rglob("*.dev*.%s.xml" % self.tgt_lang.name)))
        src_test_files = self._access_xml_seg_tags(sorted(self.dataset_directory.rglob("*.tst*.%s.xml" % self.src_lang.name)))
        tgt_test_files = self._access_xml_seg_tags(sorted(self.dataset_directory.rglob("*.tst*.%s.xml" % self.tgt_lang.name)))

        self._preprocess_store_stream_of_lines(src_train_files,
                                               self.result_directory / "train.{}".format(self.src_lang.name),
                                               self.src_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(tgt_train_files,
                                               self.result_directory / "train.{}".format(self.tgt_lang.name),
                                               self.tgt_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(src_dev_files,
                                               self.result_directory / "dev.{}".format(self.src_lang.name),
                                               self.src_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(tgt_dev_files,
                                               self.result_directory / "dev.{}".format(self.tgt_lang.name),
                                               self.tgt_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(src_test_files,
                                               self.result_directory / "test.{}".format(self.src_lang.name),
                                               self.src_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(tgt_test_files,
                                               self.result_directory / "test.{}".format(self.tgt_lang.name),
                                               self.tgt_lang, to_lower=self.to_lower)

    def _execute_wmt_prep(self):
        src_train_files = self._access_text_file_lines((self.dataset_directory / "train").rglob(
            "*.%s" % self.src_lang.name))
        tgt_train_files = self._access_text_file_lines((self.dataset_directory / "train").rglob(
            "*.%s" % self.tgt_lang.name))
        src_dev_files = self._access_xml_seg_tags(
            sorted((self.dataset_directory / "dev").rglob("*.%s.sgm" % self.src_lang.name)))
        tgt_dev_files = self._access_xml_seg_tags(
            sorted((self.dataset_directory / "dev").rglob("*.%s.sgm" % self.tgt_lang.name)))
        src_test_files = self._access_xml_seg_tags(
            sorted((self.dataset_directory / "test").rglob("*.%s.sgm" % self.src_lang.name)))
        tgt_test_files = self._access_xml_seg_tags(
            sorted((self.dataset_directory / "test").rglob("*.%s.sgm" % self.tgt_lang.name)))
        self._preprocess_store_stream_of_lines(src_train_files,
                                               self.result_directory / "train.{}".format(self.src_lang.name),
                                               self.src_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(tgt_train_files,
                                               self.result_directory / "train.{}".format(self.tgt_lang.name),
                                               self.tgt_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(src_dev_files,
                                               self.result_directory / "dev.{}".format(self.src_lang.name),
                                               self.src_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(tgt_dev_files,
                                               self.result_directory / "dev.{}".format(self.tgt_lang.name),
                                               self.tgt_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(src_test_files,
                                               self.result_directory / "test.{}".format(self.src_lang.name),
                                               self.src_lang, to_lower=self.to_lower)
        self._preprocess_store_stream_of_lines(tgt_test_files,
                                               self.result_directory / "test.{}".format(self.tgt_lang.name),
                                               self.tgt_lang, to_lower=True)

    @staticmethod
    def _access_xml_seg_tags(file_paths):
        """
        Receives a list of Path objects pointing to xml files containing "seg" tags, iterates throw the tags and returns
         the text inside those tags
        """
        for f in file_paths:
            logger.info("Pre-processing the file %s" % f.resolve())
            xml_soup = ET.parse(str(f.resolve()))
            for seg in tqdm(xml_soup.getroot().iter("seg")):
                yield seg.text

    @staticmethod
    def _access_text_file_lines(file_paths):
        """
        Receives a list of Path objects pointing to txt files, reads through them and returns their content line-by-line
        """
        for f in file_paths:
            logger.info("Pre-processing the file %s" % f.resolve())
            with f.resolve().open(encoding="utf-8") as f_opened:
                for line in tqdm(f_opened):
                    yield line

    def _preprocess_store_stream_of_lines(self, lines_stream: Iterator[str], result_file: Path, process_lang: LId,
                                          to_lower=False):
        """
        Given a stream of strings (:param lines_stream:) tokenizes each considring the rules of (:param process_lang:)
         and saves them in the :param result_file:. The file will be created if does not exists, will be overwritten
         if exists. The lines will be lower-cased if :param to_lower: flag is true.
        """
        if not result_file.exists():
            result_file.touch()
        with result_file.open(mode="w") as output_file:
            for line in lines_stream:
                if to_lower:
                    line = line.lower()
                output_file.write(" ".join(self.tokenizer.tokenize(
                    line.strip().replace("\x0b", ' ').replace("\x0c", ' ').replace("\x1c", ' ').replace("\x0d", ' ')
                        .replace(u"\u0085", ' ').replace(u"\u2028", ' '), process_lang)))
                output_file.write("\n")
                output_file.flush()


if __name__ == '__main__':
    # The single point which loads the config file passed to the script for pre-processing task
    configurations_path = sys.argv[1]
    opts = ConfigLoader(get_resource_file(configurations_path))
    logger.info('Performing the pre-processing task with the Configurations Loaded from {}:\n{}'.format(
        configurations_path, opts))
    data_preprocess = Preprocess(opts)
    data_preprocess.execute()
