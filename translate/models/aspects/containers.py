"""
This the implementation for "Syntax-Infused Information Container", a container class which loads/creates aspect set dictionaries and converts
    sentences to syntactic feature vectors to be used in syntax-infused NMT model.
"""
import os
import pickle

from readers.tokenizers import SpacyTokenizer
from models.aspects.extract_vocab import extract_linguistic_aspect_values, extract_linguistic_vocabs
from configuration import cfg, src_lan


class SyntaxInfusedInformationContainer:
    def __init__(self, bert_tokenizer):
        self.spacy_tokenizer_1 = SpacyTokenizer(src_lan, bool(cfg.lowercase_data))
        self.spacy_tokenizer_2 = SpacyTokenizer(src_lan, bool(cfg.lowercase_data))
        self.spacy_tokenizer_2.overwrite_tokenizer_with_split_tokenizer()
        self.features_list = ("f_pos", "c_pos", "subword_shape", "subword_position")
        self.bert_tokenizer = bert_tokenizer
        self.features_dict = None

    @staticmethod
    def _get_dataset_name():
        # TODO this information should be normally extracted from train.name however since in test mode train is None, we have hard coded it
        # Data comes from the "name" fields in readers.datasets.dataset classes
        # In refactoring remove this function and make it the way that data reader gives out the name of the training set even when not loading it!
        if cfg.dataset_name == "multi30k16":
            return 'm30k'
        elif cfg.dataset_name == "iwslt17":
            return 'iwslt'
        elif cfg.dataset_name == "wmt19_de_en":
            return 'wmt19_en_de'
        elif cfg.dataset_name == "wmt19_de_fr":
            return 'wmt19_de_fr'

    def load_features_dict(self, train, checkpoints_root='../.checkpoints'):
        smn = checkpoints_root + "/" + self._get_dataset_name() + "_aspect_vectors." + src_lan
        if not os.path.exists(checkpoints_root):
            os.mkdir(checkpoints_root)
        vocab_adr = smn+".vocab.pkl"
        if not os.path.exists(vocab_adr):
            assert train is not None, "syntactic vocab does not exists and training data object is empty"
            print("Starting to create linguistic vocab for for {} language ...".format(src_lan))
            ling_vocab = extract_linguistic_vocabs(train, self.bert_tokenizer, src_lan, cfg.lowercase_data)
            print("Linguistic vocab ready, persisting ...")
            pickle.dump(ling_vocab, open(vocab_adr, "wb"), protocol=4)
            print("Linguistic vocab persisted!\nDone.")
        self.features_dict = pickle.load(open(vocab_adr, "rb"), encoding="utf-8")
        for f in self.features_list:
            self.features_dict[f]["UNK_TAG"] = (len(self.features_dict[f]), 0.0)
            self.features_dict[f]["PAD_TAG"] = (len(self.features_dict[f]), 0.0)

    def _pad_tag_id(self, tag):
        return self.features_dict[tag]["PAD_TAG"][0]

    def _convert_value(self, f, value):
        if value in self.features_dict[f]:
            return self.features_dict[f][value][0]
        else:  # Test mode
            return self.features_dict[f]["UNK_TAG"][0]

    @staticmethod
    def _assure_max_len(vector, max_len):
        # this is necessary to guarantee the same number of time steps for token embeddings and feature embeddings
        # it might get a bit noisy if the generated tag sequences are not of the same size, but that's inevitable
        return vector[:max_len]

    def convert(self, sent, max_len):
        assert self.features_dict is not None, "You need to call \"load_features_dict\" first!"
        res = extract_linguistic_aspect_values(sent, self.bert_tokenizer, self.spacy_tokenizer_1, self.spacy_tokenizer_2, self.features_list)
        return {f: self._assure_max_len([self._convert_value(f, elem[f]) for elem in res] + [self._pad_tag_id(f)] * (max_len - len(res)), max_len)
                for f in self.features_list}
