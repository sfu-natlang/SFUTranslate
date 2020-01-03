import glob
import os
import io
import codecs
import spacy
import xml.etree.ElementTree as ET
from torchtext import data, datasets
from configuration import src_lan, tgt_lan, cfg, device

spacy_src = spacy.load(src_lan)
spacy_tgt = spacy.load(tgt_lan)


def src_tokenizer(text):
    if False:
        # modification of sentences to reduce the complexity of vocabulary by dealing with proper nouns separately
        # TODO
        return [tok.text if tok.pos_ != "PROPN" else cfg.propn_token for tok in spacy_src.tokenizer(text)]
    return [tok.text for tok in spacy_src.tokenizer(text)]


def tgt_tokenizer(text):
    return [tok.text for tok in spacy_tgt.tokenizer(text)]


def temp_split(x): return x.split()


if bool(cfg.debug_mode):
    src_tokenizer = temp_split
    tgt_tokenizer = temp_split


class IWSLT(datasets.TranslationDataset):
    """Do-over of the original Torchtext IWSLT Library.
    This one just does not apply filter_pred designed for length limiting to the test and dev datasets.
    The IWSLT 2016 TED talk translation task"""

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


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class MyIterator(data.Iterator):
    def __len__(self):
        return 0.0

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

# ######################################### So far configuring the dataset readers and data ######################


print("Loading the data ...")
SRC = data.Field(tokenize=src_tokenizer, lower=bool(cfg.lowercase_data), pad_token=cfg.pad_token,
                 unk_token=cfg.unk_token, include_lengths=True)
TGT = data.Field(tokenize=tgt_tokenizer, lower=bool(cfg.lowercase_data), pad_token=cfg.pad_token,
                 unk_token=cfg.unk_token, init_token=cfg.bos_token, eos_token=cfg.eos_token, include_lengths=True)
if cfg.dataset_name == "multi30k16":
    print("Debug mode: True => loading Multi30k (a smaller dataset [MinLen:1;AvgLen:12;MaxLen:40]) instead of IWSLT")
    train, val, test = datasets.translation.Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TGT))
    src_val_file_address = ".data/multi30k/val.{}".format(src_lan)
    tgt_val_file_address = ".data/multi30k/val.{}".format(tgt_lan)
    src_test_file_address = ".data/multi30k/test2016.{}".format(src_lan)
    tgt_test_file_address = ".data/multi30k/test2016.{}".format(tgt_lan)
elif cfg.dataset_name == "iwslt17":
    train, val, test = IWSLT.splits(
        filter_pred=lambda x: len(vars(x)['src']) <= cfg.max_sequence_length and len(
            vars(x)['trg']) <= cfg.max_sequence_length, exts=('.de', '.en'), fields=(SRC, TGT),
        test='IWSLT17.TED.tst2015', validation='IWSLT17.TED.dev2010', debug_mode=bool(cfg.debug_mode))
    src_val_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.dev2010.{0}-{1}.{0}".format(src_lan, tgt_lan)
    tgt_val_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.dev2010.{0}-{1}.{1}".format(src_lan, tgt_lan)
    src_test_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.tst2015.{0}-{1}.{0}".format(src_lan, tgt_lan)
    tgt_test_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.tst2015.{0}-{1}.{1}".format(src_lan, tgt_lan)
elif cfg.dataset_name == "wmt14":
    train, val, test = datasets.WMT14.splits(exts=('.de', '.en'), fields=(SRC, TGT), train='train.tok.clean.bpe.32000',
                                             validation='newstest2009.tok.bpe.32000', test='newstest2016.tok.bpe.32000')
    src_val_file_address = ".data/wmt14/newstest2009.tok.bpe.32000.{}".format(src_lan)
    tgt_val_file_address = ".data/wmt14/newstest2009.tok.bpe.32000.{}".format(tgt_lan)
    src_test_file_address = ".data/wmt14/newstest2016.tok.bpe.32000.{}".format(src_lan)
    tgt_test_file_address = ".data/wmt14/newstest2016.tok.bpe.32000.{}".format(tgt_lan)
else:
    raise ValueError("The dataset {} is not defined!".format(cfg.dataset_name))


print("Number of training examples: {}".format(len(train.examples)))
print("Number of validation examples: {}".format(len(val.examples)))
print("Number of testing examples: {}".format(len(test.examples)))

SRC.build_vocab(train, max_size=int(cfg.max_vocab_src), min_freq=int(cfg.min_freq_src))
TGT.build_vocab(train, max_size=int(cfg.max_vocab_tgt), min_freq=int(cfg.min_freq_tgt))

print("Unique tokens in source (de) vocabulary: {}".format(len(SRC.vocab)))
print("Unique tokens in target (en) vocabulary: {}".format(len(TGT.vocab)))

# Replaced the Bucket Iterator with the suggested iterator in here:
# http://nlp.seas.harvard.edu/2018/04/03/attention.html

# train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test),
#                                       batch_size=int(cfg.batch_size), device=device)

train_iter = MyIterator(train, batch_size=int(cfg.batch_size), device=device, repeat=False, train=True,
                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, shuffle=True,
                        sort_within_batch=lambda x: (len(x.src), len(x.trg)))
# val_iter = MyIterator(val, batch_size=int(cfg.batch_size), device=device, repeat=False, train=False,
#                      sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, shuffle=True,
#                      sort_within_batch=lambda x: (len(x.src), len(x.trg)))
# test_iter = MyIterator(test, batch_size=int(cfg.batch_size), device=device, repeat=False, train=False,
#                       sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, shuffle=True,
#                       sort_within_batch=lambda x: (len(x.src), len(x.trg)))
# BucketIterator keeps the order of lines which makes it easy to compare the decoded sentences with the reference
val_iter = data.BucketIterator(val, batch_size=int(cfg.batch_size), device=device, repeat=False, train=False,
                               shuffle=False, sort=False, sort_within_batch=False)
test_iter = data.BucketIterator(test, batch_size=int(cfg.batch_size), device=device, repeat=False, train=False,
                                shuffle=False, sort=False, sort_within_batch=False)

# ######################################### So far reading the dataset and preparing it #######################
