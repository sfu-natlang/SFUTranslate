"""
This is the main script which will create the aspect extractor modules. You will need a config file like the config files passed to the other trainer
 scripts of SFUTranslate to run this script. Once it is done processing the training data (source side of the train data modified in the selected
  dataset in the config file), the resulting pre-trained module will be stored in the universal checkpoints directory of the project.
"""
import os
import pickle

from torchtext import data
from tqdm import tqdm

from readers.datasets.dataset import get_dataset_from_configs
from readers.tokenizers import PTBertTokenizer
from models.aspect_extractor.extract_vocab import dataset_iterator, extract_linguistic_vocabs
from models.aspect_extractor.tester import aspect_extractor_tester
from models.aspect_extractor.trainer import aspect_extractor_trainer
from configuration import cfg, src_lan, tgt_lan


def aspect_vector_trainer(data_root='../../../.data', checkpoints_root='../../../.checkpoints', batch_size=32, H=1024, epochs=3, lr=0.05, max_norm=5,
                          scheduler_patience_steps=60, scheduler_min_lr=0.001, scheduler_decay_factor=0.9,
                          features_list=("f_pos", "c_pos", "subword_shape", "subword_position"), resolution_strategy="first"):
    # TODO add these configurations to the config file schema
    print("Starting to train the reusable heads for {} language ...".format(src_lan))
    print("Loaded the pre-created/persisted linguistic vocab dictionary ...")
    # the source data are tokenized using a split tokenizer
    SRC = data.Field(tokenize=None, lower=bool(cfg.lowercase_data), pad_token=cfg.pad_token,
                     unk_token=cfg.unk_token, include_lengths=True)
    dataset = get_dataset_from_configs(data_root, cfg.dataset_name, src_lan, tgt_lan, SRC, SRC, True, max_sequence_length=-1, sentence_count_limit=-1,
                                       debug_mode=False)
    smn = checkpoints_root + "/" + dataset.train.name + "_aspect_vectors." + src_lan
    if not os.path.exists(checkpoints_root):
        os.mkdir(checkpoints_root)
    vocab_adr = smn+".vocab.pkl"

    bert_model_name = PTBertTokenizer.get_default_model_name(src_lan, bool(cfg.lowercase_data))
    bert_tokenizer = PTBertTokenizer(src_lan, bool(cfg.lowercase_data))

    def data_itr():
        return tqdm(dataset_iterator(dataset.train, batch_size), dynamic_ncols=True)

    if not os.path.exists(vocab_adr):
        print("Starting to create linguistic vocab for for {} language ...".format(src_lan))
        ling_vocab = extract_linguistic_vocabs(dataset.train, bert_tokenizer, src_lan, cfg.lowercase_data)
        print("Linguistic vocab ready, persisting ...")
        pickle.dump(ling_vocab, open(vocab_adr, "wb"), protocol=4)
        print("Linguistic vocab persisted!\nDone.")
    ling_vocab = pickle.load(open(vocab_adr, "rb"), encoding="utf-8")
    aspect_extractor_trainer(data_itr, bert_model_name, bert_tokenizer, ling_vocab, features_list, src_lan, bool(cfg.lowercase_data), H, lr,
                             scheduler_patience_steps, scheduler_decay_factor, scheduler_min_lr, epochs, max_norm, report_every=5000,
                             save_model_name=smn, relative_sizing=False, resolution_strategy=resolution_strategy)
    print("Performing test on the training data ...")
    aspect_extractor_tester(data_itr, bert_model_name, bert_tokenizer, ling_vocab, features_list, src_lan, bool(cfg.lowercase_data),
                            load_model_name=smn, resolution_strategy=resolution_strategy, check_result_sanity=True)


if __name__ == '__main__':
    aspect_vector_trainer()
