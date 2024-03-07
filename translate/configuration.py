import sys
import re
import torch
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
match = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__)
if match:
    torch_major_version, torch_minor_version, torch_patch_version = map(int, match.groups())
    if not torch_major_version >= 1 or not torch_minor_version >= 1:
        # pack_padded_sequence forces the batches to be sorted in older versions which makes validation sloppy
        raise ValueError("You need pytorch 1.1.0+ to run this code")
else:
    raise ValueError("Could not parse torch version: " + torch.__version__)
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
    src_lan = cfg.src_lang
    tgt_lan = cfg.tgt_lang
    cfg.augment_input_with_bert_src_vectors = "aspect_augmented_transformer" in cfg.model_name or cfg.model_name == "bert_freeze_input_transformer"
    if cfg.augment_input_with_bert_src_vectors:
        assert cfg.src_tokenizer == "bert", "Using \"bert_src\" vectors must enforce bert tokenizer!"
    cfg.augment_input_with_syntax_infusion_vectors = cfg.model_name == "syntax_infused_transformer"
    if cfg.augment_input_with_syntax_infusion_vectors:  # This is for the sake of comparability to "Aspect Augmented Transformer"
        assert cfg.src_tokenizer == "bert", "Syntax Infused Transformer model should enforce bert tokenizer in source side"

