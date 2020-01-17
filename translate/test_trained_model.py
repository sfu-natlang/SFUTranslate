import torch
from configuration import cfg, device
from readers.data_provider import val_iter, src_val_file_address, tgt_val_file_address
from readers.data_provider import test_iter, src_test_file_address, tgt_test_file_address
# from models.sts import STS
from utils.evaluation import evaluate


def test_sts():
    # from readers.data_provider import SRC, TGT
    # model = STS(SRC, TGT).to(device)
    print("Loading the best trained model")
    saved_obj = torch.load(cfg.checkpoint_name, map_location=lambda storage, loc: storage)
    model = saved_obj['model'].to(device)
    model.beam_search_decoding = True
    model.beam_size = int(cfg.beam_size)
    model.beam_search_length_norm_factor = float(cfg.beam_search_length_norm_factor)
    model.beam_search_coverage_penalty_factor = float(cfg.beam_search_coverage_penalty_factor)
    # SRC = saved_obj['field_src']
    TGT = saved_obj['field_tgt']
    evaluate(val_iter, TGT, model, src_val_file_address, tgt_val_file_address, "VALIDATE")
    evaluate(test_iter, TGT, model, src_test_file_address, tgt_test_file_address, "TEST")


if __name__ == "__main__":
    test_sts()
