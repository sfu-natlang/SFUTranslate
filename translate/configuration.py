import os
import sys
import re
import torch
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
small_val = 0.0000000000000000000000001
match = re.match(r"(\d\.\d\.\d)(.*)", torch.__version__)
torch_major_version, torch_minor_version, torch_patch_version = map(int, match.group(1).split("."))
if not torch_major_version >= 1 or not torch_minor_version >= 1:
    # pack_padded_sequence forces the batches to be sorted in older versions which makes validation sloppy
    raise ValueError("You need pytorch 1.1.0+ to run this code")
if len(sys.argv) < 2:
    print("run the application with <config_file>")
    exit()
config_file = sys.argv[1]


class DotConfig:
    def __init__(self, config):
        self._cfg = config

    def __getattr__(self, k):
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v


with open(config_file, 'r') as yml_file:
    cfg = DotConfig(yaml.safe_load(yml_file))
    cfg.small_val = small_val
    src_lan = cfg.src_lang
    tgt_lan = cfg.tgt_lang
    cfg.augment_input_with_bert_src_vectors = "aspect_augmented_transformer" in cfg.model_name or cfg.model_name == "bert_freeze_input_transformer"
    if cfg.augment_input_with_bert_src_vectors:
        assert cfg.src_tokenizer == "bert", "Using \"bert_src\" vectors must enforce bert tokenizer!"
    cfg.augment_input_with_syntax_infusion_vectors = cfg.model_name == "syntax_infused_transformer"
    if cfg.augment_input_with_syntax_infusion_vectors:  # This is for the sake of comparability to "Aspect Augmented Transformer"
        assert cfg.src_tokenizer == "bert", "Syntax Infused Transformer model should enforce bert tokenizer in source side"
    # The idea from https://arxiv.org/pdf/1909.07907v1.pdf
    cfg.augment_input_with_bilingual_dict = cfg.model_name == "dictionary_fusion_transformer"
    if cfg.augment_input_with_bilingual_dict:
        print("Loading up the bilingual dictionary (assuming source in the first column) ...")
        if not os.path.isfile(cfg.bilingual_dictionary_address):
            print("The bilingual dictionary file does not exist, exiting ...")
            exit()
        bilingual_dictionary_raw = open(cfg.bilingual_dictionary_address, "r", encoding="utf-8").readlines()
        cfg.bilingual_dictionary = dict()
        for line in bilingual_dictionary_raw:
            s, t, c = line.strip().split("\t")
            if s not in cfg.bilingual_dictionary:
                cfg.bilingual_dictionary[s] = ([], [])
            if t in cfg.bilingual_dictionary[s][0]:
                raise ValueError("Duplicate key value occurrence ({}, {})".format(s, t))
            o = cfg.bilingual_dictionary[s]
            o = (o[0] + [t], o[1] + [int(c)])
            cfg.bilingual_dictionary[s] = o

        def lex(_d, _s, _t):
            if _s == _t:
                return 1.0
            elif _s in _d and _t in _d[_s][0]:
                _i = _d[_s][0].index(_t)
                return _d[_s][1][_i] / sum(_d[_s][1])
            return 0.0
        cfg.lex = lex

