import sys
import re
import torch
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
match = re.match(r"(\d\.\d\.\d)(.*)", torch.__version__)
torch_major_version, torch_minor_version, torch_patch_version = map(int, match.group(1).split("."))
if not torch_major_version >= 1 or not torch_minor_version >= 1:
    # pack_padded_sequence forces the batches to be sorted in older versions which makes validation sloppy
    raise ValueError("You need pytorch 1.1.0+ to run this code")
if len(sys.argv) < 4:
    print("run the application with <src> and <tgt>")
    exit()
src_lan = sys.argv[1]
tgt_lan = sys.argv[2]
config_file = sys.argv[3]


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
