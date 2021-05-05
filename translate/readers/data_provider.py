import torchtext
if torchtext.__version__.startswith('0.9'):
    from torchtext.legacy import data
else:
    from torchtext import data
from configuration import src_lan, tgt_lan, cfg, device
from models.aspects.containers import SyntaxInfusedInformationContainer
from readers.utils import batch_size_fn, collect_unk_stats
from readers.iterators import MyIterator, MyBucketIterator
from readers.datasets.dataset import get_dataset_from_configs
from readers.tokenizers import get_tokenizer_from_configs

src_tokenizer_obj = get_tokenizer_from_configs(cfg.src_tokenizer, src_lan, cfg.lowercase_data, debug_mode=bool(cfg.debug_mode))
tgt_tokenizer_obj = get_tokenizer_from_configs(cfg.tgt_tokenizer, tgt_lan, cfg.lowercase_data, debug_mode=bool(cfg.debug_mode))


def src_tokenizer(text):
    return src_tokenizer_obj.tokenize(text)


def tgt_tokenizer(text):
    return tgt_tokenizer_obj.tokenize(text)


def tgt_detokenizer(tokenized_list):
    return tgt_tokenizer_obj.detokenize(tokenized_list)


global max_src_in_batch, max_tgt_in_batch


class DataProvider:
    def __init__(self, SRC=None, TGT=None, load_train_data=True, root='../.data'):
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
        processed_data = get_dataset_from_configs(root, cfg.dataset_name, src_lan, tgt_lan, SRC, TGT, load_train_data, cfg.max_sequence_length,
                                                  cfg.sentence_count_limit, cfg.debug_mode)
        self.processed_data = processed_data
        if cfg.augment_input_with_syntax_infusion_vectors:
            print("Loading syntax infused tag sets ...")
            syntax_infused_container = SyntaxInfusedInformationContainer(src_tokenizer_obj)
            syntax_infused_container.load_features_dict(processed_data.train)
            src_tokenizer_obj.syntax_infused_container = syntax_infused_container
        if processed_data.train is not None:  # for testing you don't need to load train data!
            print("Number of training examples: {}".format(len(processed_data.train.examples)))
            processed_data.train.src_tokenizer = src_tokenizer_obj
        processed_data.val.src_tokenizer = src_tokenizer_obj
        for test in processed_data.test_list:
            test.src_tokenizer = src_tokenizer_obj
        print("Number of validation [set name: {}] examples: {}".format(processed_data.val.name, len(processed_data.val.examples)))
        for test in processed_data.test_list:
            print("Number of testing [set name: {}] examples: {}".format(test.name, len(test.examples)))
        if build_vocab:
            SRC.build_vocab(processed_data.train, max_size=int(cfg.max_vocab_src), min_freq=int(cfg.min_freq_src),
                            specials=[cfg.bos_token, cfg.eos_token])
            TGT.build_vocab(processed_data.train, max_size=int(cfg.max_vocab_tgt), min_freq=int(cfg.min_freq_tgt))
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
        if cfg.extract_unk_stats:
            m_unk_token = "\u26F6"
            src_unk_token = m_unk_token
            if cfg.dataset_name == "iwslt17":
                p_d = get_dataset_from_configs(root, cfg.dataset_name, src_lan, tgt_lan, SRC, TGT, True, -1, cfg.sentence_count_limit, cfg.debug_mode)
                trn = p_d.train
            else:
                trn = processed_data.train
            collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, trn, "train", processed_data.addresses.train.src,
                              processed_data.addresses.train.tgt, src_unk_token, m_unk_token)
            collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, processed_data.val, "validation", processed_data.addresses.val.src,
                              processed_data.addresses.val.tgt, src_unk_token, m_unk_token)
            for test, sa, ta in zip(processed_data.test_list, processed_data.addresses.tests.src, processed_data.addresses.tests.tgt):
                collect_unk_stats(SRC, TGT, src_tokenizer, tgt_tokenizer, test, "test", sa, ta, src_unk_token, m_unk_token)
        if processed_data.train is not None:
            self.train_iter = MyIterator(
                processed_data.train, batch_size=int(cfg.train_batch_size), device=device, repeat=False, train=True,
                sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, shuffle=True,
                sort_within_batch=lambda x: (len(x.src), len(x.trg)))
            # print("Calculating the number of training batches ...")
            # this is inefficient
            # self.size_train = len([_ for _ in self.train_iter])
            # replacing it with an upper bound estimate
            self.size_train = int(processed_data.train.max_side_total_tokens / float(cfg.train_batch_size))
        else:
            self.train_iter = None
            self.size_train = 0
        # the BucketIterator does not reorder the lines in the actual dataset file so we can compare the results of
        # the model by the actual files via reading the test/val file line-by-line skipping empty lines
        self.val_iter = MyBucketIterator(processed_data.val, batch_size=int(cfg.valid_batch_size), device=device, repeat=False,
                                         train=False, shuffle=False, sort=False, sort_within_batch=False)
        self.test_iters = [MyBucketIterator(test, batch_size=int(cfg.valid_batch_size), device=device, repeat=False,
                                            train=False, shuffle=False, sort=False, sort_within_batch=False)
                           for test in processed_data.test_list]

    def replace_fields(self, SRC, TGT):
        self.SRC = SRC
        self.TGT = TGT
