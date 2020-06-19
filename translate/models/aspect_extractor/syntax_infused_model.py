import os
import pickle
import torch
from torch import nn
from torchtext import data

from models.aspect_extractor.extract_vocab import extract_linguistic_aspect_values, extract_linguistic_vocabs
from models.transformer.model import Transformer
from models.transformer.modules import PositionalEncoding, Embeddings
from readers.tokenizers import SpacyTokenizer

from configuration import cfg, device, src_lan


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

    def convert(self, sent, max_len):
        assert self.features_dict is not None, "You need to call \"load_features_dict\" first!"
        res = extract_linguistic_aspect_values(sent, self.bert_tokenizer, self.spacy_tokenizer_1, self.spacy_tokenizer_2, self.features_list)
        return {f: [self._convert_value(f, elem[f]) for elem in res] + [self._pad_tag_id(f)] * (max_len - len(res)) for f in self.features_list}


class SyntaxInfusedSRCEmbedding(nn.Module):
    def __init__(self, src_vocab_len):
        super(SyntaxInfusedSRCEmbedding, self).__init__()
        from readers.data_provider import src_tokenizer_obj
        self.features_list = src_tokenizer_obj.syntax_infused_container.features_list
        assert len(self.features_list) > 0
        d_model = int(cfg.transformer_d_model)
        dropout = float(cfg.transformer_dropout)
        max_len = int(cfg.transformer_max_len)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len).to(device)
        self.token_embeddings = Embeddings(d_model, src_vocab_len).to(device)
        fd = src_tokenizer_obj.syntax_infused_container.features_dict
        # fd_sizes = {tag: len(fd[tag])for tag in self.features_list}
        self.syntax_embeddings = nn.ModuleList([Embeddings(int(d_model/len(self.features_list)), len(fd[tag])).to(device) for tag in self.features_list])
        self.syntax_embeddings_size = int(d_model/len(self.features_list)) * len(self.features_list)  # could be less than d_model
        self.infused_embedding_to_d_model = nn.Linear(self.syntax_embeddings_size + d_model, d_model, bias=True).to(device)

    def forward(self, input_tensor, **kwargs):
        """
        :param input_tensor: the output of vanilla transformer embedding augmented with positional encoding information
        :param kwargs: must contain the bert indexed input token ids in a tensor named "bert_src"
        :return:
        """
        syntactic_inputs = []
        for tag in self.features_list:
            lookup_tag = "si_"+tag
            if lookup_tag in kwargs and kwargs[lookup_tag] is not None:
                syntactic_inputs.append(torch.tensor(kwargs[lookup_tag], device=device).long())
            else:
                raise ValueError("The required feature tag {} information are not provided by the reader!".format(lookup_tag))
        syntactic_embedding = torch.cat([se(si) for se, si in zip(self.syntax_embeddings, syntactic_inputs)], dim=-1)
        token_embedding = self.token_embeddings(input_tensor)
        final_embedding = self.infused_embedding_to_d_model(torch.cat((syntactic_embedding, token_embedding), dim=-1))
        return self.positional_encoding(final_embedding)


class SyntaxInfusedTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(SyntaxInfusedTransformer, self).__init__(SRC, TGT)
        # TODO features_dict from SyntaxInfusedInformationContainer for test mode
        self.src_embed = SyntaxInfusedSRCEmbedding(len(SRC.vocab)).to(device)

    def init_model_params(self):
        super(SyntaxInfusedTransformer, self).init_model_params()

    def encode(self, input_tensor_with_lengths, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        """
        input_tensor, _ = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        input_mask = self.generate_src_mask(input_tensor)
        x = self.src_embed(input_tensor, **kwargs)
        for layer in self.enc_layers:
            x = layer(x, input_mask)
        return self.enc_norm(x), input_mask, input_tensor
