from torchtext import data
from configuration import src_lan, tgt_lan, cfg, device
from readers.utils import batch_size_fn, get_dataset, collect_unk_stats, MyIterator, extract_exclusive_immediate_neighbours
from readers.tokenizers import get_tokenizer_from_configs

src_tokenizer_obj = get_tokenizer_from_configs(cfg.src_tokenizer, src_lan, cfg.lowercase_data, debug_mode=bool(cfg.debug_mode))
tgt_tokenizer_obj = get_tokenizer_from_configs(cfg.tgt_tokenizer, tgt_lan, cfg.lowercase_data, debug_mode=bool(cfg.debug_mode))


def src_tokenizer(text):
    return src_tokenizer_obj.tokenize(text if cfg.src_tokenizer != "pre_trained" or not cfg.dataset_is_in_bpe else text.replace("@@ ", "").replace(" @-@ ", "-"))


def tgt_tokenizer(text):
    return tgt_tokenizer_obj.tokenize(text if cfg.tgt_tokenizer != "pre_trained" or not cfg.dataset_is_in_bpe else text.replace("@@ ", "").replace(" @-@ ", "-"))


global max_src_in_batch, max_tgt_in_batch


class DataProvider:
    def __init__(self, SRC=None, TGT=None, load_train_data=True):
        print("Loading the data ...")
        build_vocab = False
        if SRC is None or TGT is None:
            print("Creating the source and target field objects ...")
            SRC = data.Field(tokenize=src_tokenizer, lower=bool(cfg.lowercase_data), pad_token=cfg.pad_token,
                             unk_token=cfg.unk_token, include_lengths=True)
            TGT = data.Field(tokenize=tgt_tokenizer, lower=bool(cfg.lowercase_data), pad_token=cfg.pad_token,
                             unk_token=cfg.unk_token, init_token=cfg.bos_token, eos_token=cfg.eos_token, include_lengths=True)
            build_vocab = True
        else:
            print("Using the pre-loaded field objects ...")
        train, val, test_list, self.src_val_file_address, self.tgt_val_file_address, self.src_test_file_addresses, \
            self.tgt_test_file_addresses, self.src_train_file_address, self.tgt_train_file_address = get_dataset(
                src_lan, tgt_lan, SRC, TGT, load_train_data)
        if train is not None:  # for testing you don't need to load train data!
            print("Number of training examples: {}".format(len(train.examples)))
        print("Number of validation [set name: {}] examples: {}".format(val.name, len(val.examples)))
        for test in test_list:
            print("Number of testing [set name: {}] examples: {}".format(test.name, len(test.examples)))
        if build_vocab:
            SRC.build_vocab(train, max_size=int(cfg.max_vocab_src), min_freq=int(cfg.min_freq_src),
                            specials=[cfg.bos_token, cfg.eos_token])
            TGT.build_vocab(train, max_size=int(cfg.max_vocab_tgt), min_freq=int(cfg.min_freq_tgt))
        print("Unique tokens in source ({}) vocabulary: {}".format(src_lan, len(SRC.vocab)))
        print("Unique tokens in target ({}) vocabulary: {}".format(tgt_lan, len(TGT.vocab)))
        if cfg.share_vocabulary:
            print("Vocabulary sharing requested, merging the vocabulary")
            SRC.vocab.extend(TGT.vocab)
            TGT.vocab = SRC.vocab
            assert len(SRC.vocab) == len(TGT.vocab)
            print("Unique tokens in shared ({}-{}) vocabulary: {}".format(src_lan, tgt_lan, len(SRC.vocab)))
        self.TGT = TGT
        self.SRC = SRC
        self.ei_before_hyphen_list, self.ei_after_hyphen_list = extract_exclusive_immediate_neighbours(
            self.tgt_train_file_address, "-", cfg.dataset_is_in_bpe)
        if cfg.extract_unk_stats:
            m_unk_token = "\u26F6"
            src_unk_token = m_unk_token
            if cfg.dataset_name == "iwslt17_de_en":
                trn, _, _, _, _, _, _, _, _ = get_dataset(src_lan, tgt_lan, SRC, TGT, True, filter_for_max_length=False)
            else:
                trn = train
            collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, trn, "train", self.src_train_file_address,
                              self.tgt_train_file_address, src_unk_token, m_unk_token)
            collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, val, "validation", self.src_val_file_address,
                              self.tgt_val_file_address, src_unk_token, m_unk_token)
            for test, sa, ta in zip(test_list, self.src_test_file_addresses, self.tgt_test_file_addresses):
                collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, test, "test", sa, ta, src_unk_token, m_unk_token)
        if train is not None:
            self.train_iter = MyIterator(
                train, batch_size=int(cfg.train_batch_size), device=device, repeat=False, train=True,
                sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, shuffle=True,
                sort_within_batch=lambda x: (len(x.src), len(x.trg)))
            print("Calculating the number of training batches ...")
            self.size_train = len([_ for _ in self.train_iter])
        else:
            self.train_iter = None
            self.size_train = 0
        # the BucketIterator does not reorder the lines in the actual dataset file so we can compare the results of
        # the model by the actual files via reading the test/val file line-by-line skipping empty lines
        self.val_iter = data.BucketIterator(val, batch_size=int(cfg.valid_batch_size), device=device, repeat=False,
                                            train=False, shuffle=False, sort=False, sort_within_batch=False)
        self.test_iters = [data.BucketIterator(test, batch_size=int(cfg.valid_batch_size), device=device, repeat=False,
                                               train=False, shuffle=False, sort=False, sort_within_batch=False)
                           for test in test_list]

    def replace_fields(self, SRC, TGT):
        self.SRC = SRC
        self.TGT = TGT














