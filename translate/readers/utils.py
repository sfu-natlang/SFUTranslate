from torchtext import data, datasets
from configuration import cfg, device
from readers.dataset import IWSLT, WMT19DeEn
from collections import Counter
from tqdm import tqdm


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
    # TODO support multiple test sets all at once
    if cfg.dataset_name == "multi30k16":
        print("Loading Multi30k [MinLen:1;AvgLen:12;MaxLen:40]")
        train, val, test = datasets.translation.Multi30k.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                                                fields=(SRC, TGT))
        src_val_file_address = ".data/multi30k/val.{}".format(src_lan)
        tgt_val_file_address = ".data/multi30k/val.{}".format(tgt_lan)
        src_test_file_address = ".data/multi30k/test2016.{}".format(src_lan)
        tgt_test_file_address = ".data/multi30k/test2016.{}".format(tgt_lan)
        src_train_file_address = ".data/multi30k/train.{}".format(src_lan)
        tgt_train_file_address = ".data/multi30k/train.{}".format(tgt_lan)
    elif cfg.dataset_name == "iwslt17_de_en":
        dev_data = dev_data if dev_data is not None else "dev2010"
        test_data = test_data if test_data is not None else "tst2015"
        train, val, test = IWSLT.splits(
            filter_pred=lambda x: len(vars(x)['src']) <= cfg.max_sequence_length and len(
                vars(x)['trg']) <= cfg.max_sequence_length, exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
            fields=(SRC, TGT), test='IWSLT17.TED.{}'.format(test_data), validation='IWSLT17.TED.{}'.format(dev_data),
            debug_mode=bool(cfg.debug_mode))
        src_val_file_address = ".data/iwslt/de-en/IWSLT17.TED.{2}.de-en.{0}".format(src_lan, tgt_lan, dev_data)
        tgt_val_file_address = ".data/iwslt/de-en/IWSLT17.TED.{2}.de-en.{1}".format(src_lan, tgt_lan, dev_data)
        src_test_file_address = ".data/iwslt/de-en/IWSLT17.TED.{2}.de-en.{0}".format(src_lan, tgt_lan, test_data)
        tgt_test_file_address = ".data/iwslt/de-en/IWSLT17.TED.{2}.de-en.{1}".format(src_lan, tgt_lan, test_data)
        src_train_file_address = ".data/iwslt/de-en/train.de-en.{}".format(src_lan)
        tgt_train_file_address = ".data/iwslt/de-en/train.de-en.{}".format(tgt_lan)
    elif cfg.dataset_name == "wmt19_de_en" or cfg.dataset_name == "wmt19_de_en_small":
        dev_data = dev_data if dev_data is not None else "valid"
        test_data = test_data if test_data is not None else "newstest2019"
        train_data = 'train.small' if cfg.dataset_name == "wmt19_de_en_small" else "train"
        train, val, test = WMT19DeEn.splits(exts=('.{}'.format(src_lan), '.{}'.format(tgt_lan)),
                                            fields=(SRC, TGT), train=train_data,
                                            validation='{}-ende.bpe'.format(dev_data),
                                            test='{}-ende.bpe'.format(test_data))
        if dev_data == "valid":
            src_val_file_address = ".data/wmt19_en_de/valid.{}".format(src_lan)
            tgt_val_file_address = ".data/wmt19_en_de/valid.{}".format(tgt_lan)
        else:
            src_val_file_address = ".data/wmt19_en_de/{}-original-ende.{}".format(dev_data, src_lan)
            tgt_val_file_address = ".data/wmt19_en_de/{}-original-ende.{}".format(dev_data, tgt_lan)
        src_test_file_address = ".data/wmt19_en_de/{}-original-ende.{}".format(test_data, src_lan)
        tgt_test_file_address = ".data/wmt19_en_de/{}-original-ende.{}".format(test_data, tgt_lan)
        src_train_file_address = ".data/wmt19_en_de/{}.{}".format(train_data, src_lan)
        tgt_train_file_address = ".data/wmt19_en_de/{}.{}".format(train_data, tgt_lan)
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
        src_train_file_address = ".data/wmt14/train.tok.clean.bpe.32000.{}".format(src_lan)
        tgt_train_file_address = ".data/wmt14/train.tok.clean.bpe.32000.{}".format(tgt_lan)
    else:
        raise ValueError("The dataset {} is not defined in torchtext or SFUTranslate!".format(cfg.dataset_name))

    return train, val, test, src_val_file_address, tgt_val_file_address, src_test_file_address, tgt_test_file_address, \
           src_train_file_address, tgt_train_file_address


def collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, dt_raw, dataset_name,
                      src_file_adr, tgt_file_adr, src_unk_token, m_unk_token="\u26F6"):
    from utils.evaluation import convert_target_batch_back
    from readers.sequence_alignment import extract_monotonic_sequence_to_sequence_alignment

    def _get_next_line(file_1_iter, file_2_iter):
        for l1, l2 in zip(file_1_iter, file_2_iter):
            l1 = l1.strip()
            l2 = l2.strip()
            if not len(l1) or not len(l2):
                continue
            yield l1, l2
    dt_iter = data.BucketIterator(dt_raw, batch_size=1, device=device, repeat=False, train=False, shuffle=False,
                                  sort=False, sort_within_batch=False)
    # BucketIterator will ignore the lines one side of which is an empty line
    to_lower = bool(cfg.lowercase_data)
    src_originals = iter(open(src_file_adr, "r"))
    tgt_originals = iter(open(tgt_file_adr, "r"))
    src_unk_cnt = Counter()
    trg_unk_cnt = Counter()
    src_cnt = Counter()
    trg_cnt = Counter()
    print("Collecting UNK token/type statistics ...")
    for dt, (s, t) in tqdm(zip(dt_iter, _get_next_line(src_originals, tgt_originals))):
        dt_src_proc = convert_target_batch_back(dt.src[0], SRC)[0].replace(cfg.unk_token, m_unk_token).replace(" ##","")
        dt_src_orig = s.lower() if to_lower else s
        dt_trg_proc = convert_target_batch_back(dt.trg[0], TGT)[0].replace(cfg.unk_token, m_unk_token).replace(" ##","")
        dt_trg_orig = t.lower() if to_lower else t
        dt_src = src_tokenizer(dt_src_proc)
        dt_trg = tgt_tokenizer(dt_trg_proc)
        st = src_tokenizer(dt_src_orig)
        tt = tgt_tokenizer(dt_trg_orig)
        src_a = extract_monotonic_sequence_to_sequence_alignment(dt_src, st)
        trg_a = extract_monotonic_sequence_to_sequence_alignment(dt_trg, tt)
        ss_idx = 0
        for s_token, f in zip(dt_src, src_a):
            for i in range(f):
                ss_token = st[ss_idx+i]  # st will never contain [UNK] token.
                if src_unk_token == s_token:
                    src_unk_cnt[ss_token] += 1.0
                src_cnt[ss_token] += 1.0
            ss_idx += f
        tt_idx = 0
        for t_token, f in zip(dt_trg, trg_a):
            for i in range(f):
                tt_token = tt[tt_idx+i]  # tt will never contain [UNK] token.
                if m_unk_token == t_token:
                    trg_unk_cnt[tt_token] += 1.0
                trg_cnt[tt_token] += 1.0
            tt_idx += f

    print("UNK percentage of source vocabulary types over {} data: {:.2f}% ".format(dataset_name, float(
        len(src_unk_cnt)) * 100.0 / len(src_cnt)))
    print("UNK percentage of target vocabulary types over {} data: {:.2f}% ".format(dataset_name, float(
        len(trg_unk_cnt)) * 100.0 / len(trg_cnt)))
    print("UNK percentage of source vocabulary tokens over {} data: {:.2f}% ".format(dataset_name, float(
        sum(src_unk_cnt.values())) * 100.0 / sum(src_cnt.values())))
    print("UNK percentage of target vocabulary tokens over {} data: {:.2f}% ".format(dataset_name, float(
        sum(trg_unk_cnt.values())) * 100.0 / sum(trg_cnt.values())))
