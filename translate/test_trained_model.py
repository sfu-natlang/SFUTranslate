import torch
from configuration import cfg, device
from reader import val_iter, src_val_file_address, tgt_val_file_address
from reader import test_iter, src_test_file_address, tgt_test_file_address
# from sts_model import STS
from evaluation_utils import evaluate


def test_sts():
    # from reader import SRC, TGT
    # model = STS(SRC, TGT).to(device)
    print("Loading the best trained model")
    saved_obj = torch.load(cfg.checkpoint_name, map_location=lambda storage, loc: storage)
    model = saved_obj['model'].to(device)
    # it might not correctly overwrite the vocabulary objects
    # SRC = saved_obj['field_src']
    TGT = saved_obj['field_tgt']
    evaluate(val_iter, TGT, model, src_val_file_address, tgt_val_file_address, "VALIDATE")
    evaluate(test_iter, TGT, model, src_test_file_address, tgt_test_file_address, "TEST")


if __name__ == "__main__":
    test_sts()
