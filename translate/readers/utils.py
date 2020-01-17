from torchtext import data, datasets
from configuration import cfg
from readers.dataset import IWSLT


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
    """
    The customized torchtext iterator suggested in https://nlp.seas.harvard.edu/2018/04/03/attention.html
    The iterator is meant to speed up the training by token-wise batching
    """
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


def get_dataset(src_lan, tgt_lan, SRC: data.Field, TGT: data.Field, dev_data=None, test_data=None):
    if cfg.dataset_name == "multi30k16":
        print("Loading Multi30k (a smaller dataset [MinLen:1;AvgLen:12;MaxLen:40]) instead of IWSLT")
        train, val, test = datasets.translation.Multi30k.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                                                fields=(SRC, TGT))
        src_val_file_address = ".data/multi30k/val.{}".format(src_lan)
        tgt_val_file_address = ".data/multi30k/val.{}".format(tgt_lan)
        src_test_file_address = ".data/multi30k/test2016.{}".format(src_lan)
        tgt_test_file_address = ".data/multi30k/test2016.{}".format(tgt_lan)
    elif cfg.dataset_name == "iwslt17":
        dev_data = dev_data if dev_data is not None else "dev2010"
        test_data = test_data if test_data is not None else "tst2015"
        train, val, test = IWSLT.splits(
            filter_pred=lambda x: len(vars(x)['src']) <= cfg.max_sequence_length and len(
                vars(x)['trg']) <= cfg.max_sequence_length, exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
            fields=(SRC, TGT), test='IWSLT17.TED.{}'.format(test_data), validation='IWSLT17.TED.{}'.format(dev_data),
            debug_mode=bool(cfg.debug_mode))
        src_val_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.{2}.{0}-{1}.{0}".format(src_lan, tgt_lan, dev_data)
        tgt_val_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.{2}.{0}-{1}.{1}".format(src_lan, tgt_lan, dev_data)
        src_test_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.{2}.{0}-{1}.{0}".format(src_lan, tgt_lan, test_data)
        tgt_test_file_address = ".data/iwslt/{0}-{1}/IWSLT17.TED.{2}.{0}-{1}.{1}".format(src_lan, tgt_lan, test_data)
    elif cfg.dataset_name == "wmt14":
        dev_data = dev_data if dev_data is not None else "newstest2009"
        test_data = test_data if test_data is not None else "newstest2016"
        train, val, test = datasets.WMT14.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                                 fields=(SRC, TGT), train='train.tok.clean.bpe.32000',
                                                 validation='{}.tok.bpe.32000'.format(dev_data),
                                                 test='{}.tok.bpe.32000'.format(test_data))
        src_val_file_address = ".data/wmt14/{}.tok.bpe.32000.{}".format(dev_data, src_lan)
        tgt_val_file_address = ".data/wmt14/{}.tok.bpe.32000.{}".format(dev_data, tgt_lan)
        src_test_file_address = ".data/wmt14/{}.tok.bpe.32000.{}".format(test_data, src_lan)
        tgt_test_file_address = ".data/wmt14/{}.tok.bpe.32000.{}".format(test_data, tgt_lan)
    else:
        raise ValueError("The dataset {} is not defined in torchtext or SFUTranslate!".format(cfg.dataset_name))

    return train, val, test, src_val_file_address, tgt_val_file_address, src_test_file_address, tgt_test_file_address
