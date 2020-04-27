import codecs
import sys
import spacy
from torchtext import data, datasets
from readers.dataset import IWSLT

dataset_name = sys.argv[1]
src_lan, tgt_lan = sys.argv[2], sys.argv[3]
spacy_src = spacy.load(src_lan)
spacy_tgt = spacy.load(tgt_lan)


def src_tokenizer(text):
    return [tok.text for tok in spacy_src.tokenizer(text)]


def tgt_tokenizer(text):
    return [tok.text for tok in spacy_tgt.tokenizer(text)]


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
        train, val, *test_list = IWSLT.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)), fields=(SRC, TGT))
    elif dataset_name == "wmt14":
        train, val, test_single = datasets.WMT14.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                                        fields=(SRC, TGT), train='train.tok.clean.bpe.32000')
        test_list = [test_single]
    else:
        raise ValueError("The dataset {} is not defined!".format(dataset_name))

    print("Number of training examples: {}".format(len(train.examples)))
    print("Number of validation examples: {}".format(len(val.examples)))
    for test in test_list:
        print("Number of testing [set name: {}] examples: {}".format(test.name, len(test.examples)))

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
