import glob
import os
import io
import codecs
import spacy
import xml.etree.ElementTree as ET
from torchtext import data, datasets
from configuration import src_lan, tgt_lan, cfg, device
from readers.utils import *

spacy_src = spacy.load(src_lan)
spacy_tgt = spacy.load(tgt_lan)


def src_tokenizer(text):
    return [tok.text for tok in spacy_src.tokenizer(text)]


def tgt_tokenizer(text):
    return [tok.text for tok in spacy_tgt.tokenizer(text)]


def temp_split(x): return x.split()


if bool(cfg.debug_mode):
    src_tokenizer = temp_split
    tgt_tokenizer = temp_split

global max_src_in_batch, max_tgt_in_batch


print("Loading the data ...")
SRC = data.Field(tokenize=src_tokenizer, lower=bool(cfg.lowercase_data), pad_token=cfg.pad_token,
                 unk_token=cfg.unk_token, include_lengths=True)
TGT = data.Field(tokenize=tgt_tokenizer, lower=bool(cfg.lowercase_data), pad_token=cfg.pad_token,
                 unk_token=cfg.unk_token, init_token=cfg.bos_token, eos_token=cfg.eos_token, include_lengths=True)
# HINT: different test and validation set names must be read from config to be able to test on different test data
train, val, test, src_val_file_address, tgt_val_file_address, src_test_file_address, tgt_test_file_address, \
    src_train_file_address, tgt_train_file_address = get_dataset(src_lan, tgt_lan, SRC, TGT)

print("Number of training examples: {}".format(len(train.examples)))
print("Number of validation examples: {}".format(len(val.examples)))
print("Number of testing examples: {}".format(len(test.examples)))

SRC.build_vocab(train, max_size=int(cfg.max_vocab_src), min_freq=int(cfg.min_freq_src),
                specials=[cfg.bos_token, cfg.eos_token])
TGT.build_vocab(train, max_size=int(cfg.max_vocab_tgt), min_freq=int(cfg.min_freq_tgt))

print("Unique tokens in source ({}) vocabulary: {}".format(src_lan, len(SRC.vocab)))
print("Unique tokens in target ({}) vocabulary: {}".format(tgt_lan, len(TGT.vocab)))

if cfg.extract_unk_stats:
    m_unk_token = "\u26F6"
    src_unk_token = m_unk_token
    if cfg.dataset_name == "iwslt17_de_en":
        trn, _, _, _, _, _, _, _, _ = get_dataset(src_lan, tgt_lan, SRC, TGT, filter_for_max_length=False)
    else:
        trn = train
    collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, trn, "train", src_train_file_address,
                      tgt_train_file_address, src_unk_token, m_unk_token)
    collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, val, "validation", src_val_file_address,
                      tgt_val_file_address, src_unk_token, m_unk_token)
    collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, test, "test", src_test_file_address,
                      tgt_test_file_address, src_unk_token, m_unk_token)

train_iter = MyIterator(train, batch_size=int(cfg.train_batch_size), device=device, repeat=False, train=True,
                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, shuffle=True,
                        sort_within_batch=lambda x: (len(x.src), len(x.trg)))
# the BucketIterator does not reorder the lines in the actual dataset file so we can compare the results of the model
# by the actual files via reading the test/val file line-by-line
val_iter = data.BucketIterator(val, batch_size=int(cfg.valid_batch_size), device=device, repeat=False, train=False,
                               shuffle=False, sort=False, sort_within_batch=False)
test_iter = data.BucketIterator(test, batch_size=int(cfg.valid_batch_size), device=device, repeat=False, train=False,
                                shuffle=False, sort=False, sort_within_batch=False)
