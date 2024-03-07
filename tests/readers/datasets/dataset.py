# -*- coding: utf-8 -*-
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import sys
import os
import unittest

from readers.datasets.dataset import M30k, IWSLT, WMT19DeEn, WMT19DeFr
from readers.data.field import Field


def src_tokenizer(text):
    return text.split()


def tgt_tokenizer(text):
    return text.split()


class TestDatasetLoaders(unittest.TestCase):
    """Data Loader Test Cases."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.maxDiff = 1300
        # data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../translate/.data/'))
        # if not os.path.exists(data_dir):
        #    raise ValueError("The data directory is not accessible under {}".format(data_dir))
        cls.m30k_languages = ['fr', 'en', 'de']
        cls.iwslt_languages = ['ar', 'de', 'fr', 'ja', 'ko', 'zh']
        cls.SRC = Field(tokenize=src_tokenizer, lower=True, pad_token="<pad>", unk_token="<unk>", include_lengths=True)
        cls.TGT = Field(tokenize=tgt_tokenizer, lower=True, pad_token="<pad>", unk_token="<unk>", init_token="<bos>",
                             eos_token="<eos>", include_lengths=True)
        cls.root_dir = '../../../.data'

    def test_wmt_de_en_loader(self):
        lan_src = 'de'
        lan_tgt = 'en'
        print("Loading {}-{} WMT dataset".format(lan_src, lan_tgt))
        WMT19DeEn.splits(exts=('.{}'.format(lan_src), '.{}'.format(lan_tgt)), fields=(self.SRC, self.TGT), sentence_count_limit=1000, root=self.root_dir)
        print("Loading {}-{} WMT dataset".format(lan_tgt, lan_src))
        WMT19DeEn.splits(exts=('.{}'.format(lan_tgt), '.{}'.format(lan_src)), fields=(self.SRC, self.TGT), sentence_count_limit=1000, root=self.root_dir)

    def test_wmt_de_fr_loader(self):
        lan_src = 'de'
        lan_tgt = 'fr'
        print("Loading {}-{} WMT dataset".format(lan_src, lan_tgt))
        WMT19DeFr.splits(exts=('.{}'.format(lan_src), '.{}'.format(lan_tgt)), fields=(self.SRC, self.TGT), sentence_count_limit=1000, root=self.root_dir)
        print("Loading {}-{} WMT dataset".format(lan_tgt, lan_src))
        WMT19DeFr.splits(exts=('.{}'.format(lan_tgt), '.{}'.format(lan_src)), fields=(self.SRC, self.TGT), sentence_count_limit=1000, root=self.root_dir)

    def test_iwslt_loader(self):
        lan_tgt = 'en'
        for lan_src in [self.iwslt_languages[0]]:
            if lan_src == lan_tgt:
                continue
            print("Loading {}-{} IWSLT dataset".format(lan_src, lan_tgt))
            IWSLT.splits(exts=('.{}'.format(lan_src), '.{}'.format(lan_tgt)), fields=(self.SRC, self.TGT), root=self.root_dir)

    def test_m30k_loader(self):
        for lan_src in self.m30k_languages:
            for lan_tgt in self.m30k_languages:
                if lan_src == lan_tgt:
                    continue
                print("Loading {}-{} M30k dataset".format(lan_src, lan_tgt))
                M30k.splits(exts=('.{}'.format(lan_src), '.{}'.format(lan_tgt)), fields=(self.SRC, self.TGT), root=self.root_dir)


if __name__ == '__main__':
    unittest.main()
