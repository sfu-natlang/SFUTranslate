import codecs
import io
import os
import sys
import glob
import xml.etree.ElementTree as ET
import spacy
from torchtext import data, datasets

dataset_name = sys.argv[1]
src_lan, tgt_lan = sys.argv[2], sys.argv[3]
spacy_src = spacy.load(src_lan)
spacy_tgt = spacy.load(tgt_lan)


def src_tokenizer(text):
    return [tok.text for tok in spacy_src.tokenizer(text)]


def tgt_tokenizer(text):
    return [tok.text for tok in spacy_tgt.tokenizer(text)]


class IWSLT(datasets.TranslationDataset):
    """Do-over of the original Torchtext IWSLT Library.
    This one just does not apply filter_pred designed for length limiting to the test and dev datasets.
    The IWSLT 2017 TED talk translation task"""

    # base_url = 'https://wit3.fbk.eu/archive/2016-01//texts/{}/{}/{}.tgz'
    base_url = 'https://wit3.fbk.eu/archive/2017-01-trnted/texts/{}/{}/{}.tgz'
    name = 'iwslt'
    base_dirname = '{}-{}'

    @classmethod
    def splits(cls, exts, fields, root='.data',
               train='train', validation='IWSLT17.TED.dev2010',
               test='IWSLT17.TED.tst2015', **kwargs):
        debug_mode = False
        if "debug_mode" in kwargs:
            debug_mode = kwargs["debug_mode"]
            del kwargs["debug_mode"]
        cls.dirname = cls.base_dirname.format(exts[0][1:], exts[1][1:])
        cls.urls = [cls.base_url.format(exts[0][1:], exts[1][1:], cls.dirname)]
        check = os.path.join(root, cls.name, cls.dirname)
        path = cls.download(root, check=check)

        train = '.'.join([train, cls.dirname])
        validation = '.'.join([validation, cls.dirname])
        if test is not None:
            test = '.'.join([test, cls.dirname])

        if not os.path.exists(os.path.join(path, train) + exts[0]):
            cls.clean(path)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        # Here is the line that have been added.
        if not debug_mode:
            kwargs['filter_pred'] = None
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @staticmethod
    def clean(path):
        for f_xml in glob.iglob(os.path.join(path, '*.xml')):
            print(f_xml)
            f_txt = os.path.splitext(f_xml)[0]
            with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt:
                root = ET.parse(f_xml).getroot()[0]
                for doc in root.findall('doc'):
                    for e in doc.findall('seg'):
                        fd_txt.write(e.text.strip() + '\n')

        xml_tags = ['<url', '<keywords', '<talkid', '<description',
                    '<reviewer', '<translator', '<title', '<speaker']
        for f_orig in glob.iglob(os.path.join(path, 'train.tags*')):
            print(f_orig)
            f_txt = f_orig.replace('.tags', '')
            with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt, \
                    io.open(f_orig, mode='r', encoding='utf-8') as fd_orig:
                for l in fd_orig:
                    if not any(tag in l for tag in xml_tags):
                        fd_txt.write(l.strip() + '\n')


if __name__ == '__main__':
    print("Loading the data ...")

    SRC = data.Field(tokenize=src_tokenizer, lower=True, pad_token='<pad>', unk_token='<unk>',
                     init_token='<bos>', eos_token='<eos>', include_lengths=True)
    TGT = data.Field(tokenize=tgt_tokenizer, lower=True, pad_token='<pad>', unk_token='<unk>',
                     init_token='<bos>', eos_token='<eos>', include_lengths=True)
    if dataset_name == "multi30k16":
        train, val, test = datasets.translation.Multi30k.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                                                fields=(SRC, TGT))
    elif dataset_name == "iwslt17":
        train, val, test = IWSLT.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT))
    elif dataset_name == "wmt14":
        train, val, test = datasets.WMT14.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                                 fields=(SRC, TGT), train='train.tok.clean.bpe.32000')
    else:
        raise ValueError("The dataset {} is not defined!".format(dataset_name))
    print("Number of training examples: {}".format(len(train.examples)))
    print("Number of validation examples: {}".format(len(val.examples)))
    print("Number of testing examples: {}".format(len(test.examples)))

    SRC.build_vocab(train, max_size=2000000, min_freq=0)
    TGT.build_vocab(train, max_size=2000000, min_freq=0)
    src_vocab = set(SRC.vocab.stoi.keys())
    tgt_vocab = set(TGT.vocab.stoi.keys())
    specials = list(src_vocab.intersection(tgt_vocab))

    with codecs.open("../resources/{}_{}_{}.common.vocab".format(dataset_name, src_lan, tgt_lan),
                     "w", encoding="utf-8") as result:
        for word in specials:
            result.write(word + "\n")
    # SRC.build_vocab(train, max_size=25000, min_freq=1, specials=specials)
    # TGT.build_vocab(train, max_size=25000, min_freq=1, specials=specials)
    # print(SRC.vocab.stoi[word] == TGT.vocab.stoi[word])
    # print(word)
    # print(len(SRC))
