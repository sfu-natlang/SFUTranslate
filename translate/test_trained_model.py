import time
import torch
from configuration import cfg, device
from readers.data_provider import DataProvider
from utils.evaluation import evaluate
# To avoid the annoying UserWarnings of torchtext
# Remove this once the next version of torchtext is available
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def test_trained_model():
    print("Loading the best trained model")
    saved_obj = torch.load("../.checkpoints/"+cfg.checkpoint_name, map_location=lambda storage, loc: storage)
    model = saved_obj['model'].to(device)
    if 'training_evaluation_results' in saved_obj:
        print("Comma separated greedy decoding validation set BlueP1 scores collected during training:\n\t ===> {}".format(
            ",".join(["{:.2f}".format(x) for x in saved_obj['training_evaluation_results']])))
    model.beam_search_decoding = True
    model.beam_size = int(cfg.beam_size)
    model.beam_search_length_norm_factor = float(cfg.beam_search_length_norm_factor)
    model.beam_search_coverage_penalty_factor = float(cfg.beam_search_coverage_penalty_factor)
    SRC = saved_obj['field_src']
    TGT = saved_obj['field_tgt']
    print("Model loaded, total number of parameters: {}".format(sum([p.numel() for p in model.parameters()])))
    dp = DataProvider(SRC, TGT, load_train_data=False)
    nuance = str(int(time.time()))
    evaluate(dp.val_iter, dp, model, dp.processed_data.addresses.val.src, dp.processed_data.addresses.val.tgt,
             dp.processed_data.addresses.val.src_sgm, dp.processed_data.addresses.val.tgt_sgm,
             "VALID.{}".format(dp.val_iter.dataset.name), save_decoded_sentences=True, nuance=nuance)
    for test_iter, s, t, s_sgm, t_sgm in zip(dp.test_iters, dp.processed_data.addresses.tests.src, dp.processed_data.addresses.tests.tgt,
                                             dp.processed_data.addresses.tests.src_sgm, dp.processed_data.addresses.tests.tgt_sgm):
        evaluate(test_iter, dp, model, s, t, s_sgm, t_sgm, "TEST.{}".format(test_iter.dataset.name), save_decoded_sentences=True, nuance=nuance)


if __name__ == "__main__":
    test_trained_model()
