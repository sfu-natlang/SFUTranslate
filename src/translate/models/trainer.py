# !/usr/bin/bash
"""
The starting point of the project. Ideally we would not need to change this class for performing different models.
You can run this script using the following bash script [The config file "default.yaml" will be looked up
 from /path/to/SFUTranslate/resources]:
####################################################################
#!/usr/bin/env bash
cd /path/to/SFUTranslate/src && python -m translate.models.trainer default.yaml
####################################################################
"""
import math
from tqdm import tqdm
import sys
from typing import Type

from translate.configs.loader import ConfigLoader
from translate.configs.utils import get_resource_file
from translate.models.RNN.estimator import Estimator, StatCollector
from translate.models.RNN.seq2seq import SequenceToSequence
from translate.models.backend.padder import get_padding_batch_loader
from translate.models.backend.utils import device
from translate.readers.constants import ReaderType
from translate.readers.datareader import AbsDatasetReader
from translate.readers.dummydata import ReverseCopyDataset, SimpleGrammerLMDataset
from translate.logging.utils import logger

__author__ = "Hassan S. Shavarani"


def prepare_datasets(configs: ConfigLoader, dataset_class: Type[AbsDatasetReader]):
    """
    The Dataset provider method for train, test and dev datasets
    :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
    :param dataset_class: the classname of the intended Dataset reader. 
    """
    train_ = dataset_class(configs, ReaderType.TRAIN)
    test_ = dataset_class(configs, ReaderType.TEST)
    dev_ = dataset_class(configs, ReaderType.DEV)
    return train_, test_, dev_


if __name__ == '__main__':
    # The single point which loads the config file passed to the script
    opts = ConfigLoader(get_resource_file(sys.argv[1]))
    dataset_type = opts.get("reader.dataset.type", must_exist=True)
    epochs = opts.get("trainer.optimizer.epochs", must_exist=True)
    model_type = opts.get("trainer.model.type")
    # to support more dataset types you need to extend this list
    if dataset_type == "dummy_s2s":
        train, test, dev = prepare_datasets(opts, ReverseCopyDataset)
    elif dataset_type == "dummy_lm":
        train, test, dev = prepare_datasets(opts, SimpleGrammerLMDataset)
    elif dataset_type == "parallel":
        train, test, dev = prepare_datasets(opts, None)
    else:
        raise NotImplementedError

    if model_type == "seq2seq":
        model = SequenceToSequence(opts, train).to(device)
    else:
        raise NotImplementedError
    estimator = Estimator(opts, model)
    stat_collector = StatCollector()
    # the value which is used for performing the dev set evaluation steps
    print_every = int(0.25 * int(math.ceil(float(len(train) / float(model.batch_size)))))

    for epoch in range(epochs):
        logger.info("Epoch {}/{} begins ...".format(epoch + 1, epochs))
        iter_ = 0
        train.allocate()
        itr_handler = tqdm(get_padding_batch_loader(train, model.batch_size), ncols=100,
                           desc="[E {}/{}]-[B {}]-[L {}]-#Batches Processed".format(
                               epoch + 1, epochs, model.batch_size, 0.0),
                           total=math.ceil(len(train) / model.batch_size))
        for input_tensor_batch, target_tensor_batch in itr_handler:
            iter_ += 1
            loss_value, decoded_word_ids = estimator.step(input_tensor_batch, target_tensor_batch)
            stat_collector.update(1.0, loss_value, ReaderType.TRAIN)
            itr_handler.set_description("[E {}/{}]-[B {}]-[TL {:.3f} DL {:.3f} DS {:.3f}]-#Batches Processed"
                                        .format(epoch + 1, epochs, model.batch_size, stat_collector.train_loss,
                                                stat_collector.dev_loss, stat_collector.dev_score))
            if iter_ % print_every == 0:
                dev.allocate()
                ref_sample, hyp_sample = "", ""
                for batch_i_tensor, batch_t_tensor in get_padding_batch_loader(dev, model.batch_size):
                    dev_loss_value, dev_decoded_word_ids = estimator.step_no_grad(batch_i_tensor, batch_t_tensor)
                    bleu_score, ref_sample, hyp_sample = train.compute_bleu(target_tensor_batch, dev_decoded_word_ids,
                                                                            ref_is_tensor=True, hyp_is_tensor=False)
                    stat_collector.update(bleu_score, dev_loss_value, ReaderType.DEV)
                print("", end='\n', file=sys.stderr)
                logger.info(u"Sample: E=\"{}\", P=\"{}\"\n".format(ref_sample, hyp_sample))
                dev.deallocate()
                # TODO save the best model in here
                # TODO add early stopping criteria
        print("\n", end='\n', file=sys.stderr)
        train.deallocate()
        # TODO add test step in here
