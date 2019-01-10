"""
The starting point of the project. Ideally we would not need to change this class for performing different models.
You can run this script using the following bash script [The config file "dummy.yaml" will be looked up
 from /path/to/SFUTranslate/resources]:
####################################################################
#!/usr/bin/env bash
cd /path/to/SFUTranslate/src && python -m translate.models.trainer dummy.yaml
####################################################################
"""
import math
from tqdm import tqdm
import sys
from typing import Type

from translate.configs.loader import ConfigLoader
from translate.configs.utils import get_resource_file
from translate.learning.estimator import Estimator, StatCollector
from translate.learning.modelling import AbsCompleteModel
from translate.learning.models.rnn.lm import RNNLM
from translate.learning.models.rnn.seq2seq import SequenceToSequence
from translate.learning.models.transformer.transducer import Transformer
from translate.backend.padder import get_padding_batch_loader
from translate.backend.utils import device
from translate.readers.constants import ReaderType
from translate.readers.datareader import AbsDatasetReader
from translate.readers.datawrapper import TransformerReaderWrapper
from translate.readers.paralleldata import ParallelDataReader
from translate.readers.dummydata import ReverseCopyDataset, SimpleGrammerLMDataset
from translate.logging.utils import logger

__author__ = "Hassan S. Shavarani"


def perform_no_grad_dataset_iteration(dataset: AbsDatasetReader, model_estimator: Estimator,
                                      complete_model: Type[AbsCompleteModel], stats_collector: StatCollector):
    """
    The function which goes over the passed :param dataset: and computes the validation scores for each batch in it
     using :param model_estimator: and the :param complete_model: sent to it. Validation results are then updated in the
      passed :param stats_collector: instance. 
    """
    dataset.allocate()
    dev_sample = ""
    for dev_values in get_padding_batch_loader(dataset, complete_model.batch_size):
        dev_score, dev_loss, dev_sample = complete_model.validate_instance(*model_estimator.step_no_grad(*dev_values),
                                                                           *dev_values)
        stats_collector.update(dev_score, dev_loss, dataset.reader_type)
    print("", end='\n', file=sys.stderr)
    logger.info(u"Sample: {}".format(dev_sample))
    dataset.deallocate()


def prepare_datasets(configs: ConfigLoader, dataset_class: Type[AbsDatasetReader]):
    """
    The Dataset provider method for train, test and dev datasets
    :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
    :param dataset_class: the classname of the intended Dataset reader. 
    """
    train_ = dataset_class(configs, ReaderType.TRAIN)
    test_ = dataset_class(configs, ReaderType.TEST, shared_reader_data=train_.get_sharable_data())
    dev_ = dataset_class(configs, ReaderType.DEV, shared_reader_data=train_.get_sharable_data())
    return train_, test_, dev_


if __name__ == '__main__':
    # The single point which loads the config file passed to the script
    configurations_path = sys.argv[1]
    opts = ConfigLoader(get_resource_file(configurations_path))
    logger.info('Configurations Loaded from {}:\n{}'.format(configurations_path, opts))
    dataset_type = opts.get("reader.dataset.type", must_exist=True)
    epochs = opts.get("trainer.optimizer.epochs", must_exist=True)
    save_best_models = opts.get("trainer.optimizer.save_best_models", False)
    early_stopping_loss = opts.get("trainer.optimizer.early_stopping_loss", 0.01)
    model_type = opts.get("trainer.model.type")
    # to support more dataset types you need to extend this list
    if dataset_type == "dummy_parallel":
        train, test, dev = prepare_datasets(opts, ReverseCopyDataset)
    elif dataset_type == "dummy_lm":
        train, test, dev = prepare_datasets(opts, SimpleGrammerLMDataset)
    elif dataset_type == "parallel":
        train, test, dev = prepare_datasets(opts, ParallelDataReader)
    else:
        raise NotImplementedError

    if model_type == "seq2seq":
        model = SequenceToSequence(opts, train).to(device)
    elif model_type == "rnnlm":
        model = RNNLM(opts, train).to(device)
    elif model_type == "transformer":
        train, test, dev = TransformerReaderWrapper(train), TransformerReaderWrapper(test), TransformerReaderWrapper(
            dev)
        model = Transformer(opts, train).to(device)
    else:
        raise NotImplementedError
    estimator = Estimator(opts, model)
    stat_collector = StatCollector()
    # the value which is used for performing the dev set evaluation steps
    print_every = math.ceil(0.25 * int(math.ceil(float(len(train) / float(model.batch_size)))))
    best_saved_model_path = opts.get("trainer.model.best_model_path", None)
    early_stopping = False
    if epochs > 0:
        logger.info("Starting the training procedure ...")
        if best_saved_model_path is not None:
            logger.info("Loading the best previously trained model from {} to continue the training over it".format(
                best_saved_model_path))
            model = estimator.load_checkpoint(best_saved_model_path)
    else:
        logger.info("Skipping the training part [#epochs = 0]")
    for epoch in range(epochs):
        if early_stopping:
            logger.info("Early stopping criteria fulfilled, stopping the training ...")
            break
        iter_ = 0
        logger.info("Epoch {}/{} begins ...".format(epoch + 1, epochs))
        train.allocate()
        itr_handler = tqdm(get_padding_batch_loader(train, model.batch_size), ncols=100,
                           total=math.ceil(len(train) / model.batch_size))
        train.set_iter_log_handler(itr_handler.set_description)
        dev.set_iter_log_handler(itr_handler.set_description)
        for train_batch in itr_handler:
            iter_ += 1
            loss_value, decoded_word_ids = estimator.step(*train_batch)
            stat_collector.update(1.0, loss_value, train.reader_type)
            itr_handler.set_description("[E {}/{}]-[B {}]-[TL {:.3f} DL {:.3f} DS {:.3f}]-#Batches Processed"
                                        .format(epoch + 1, epochs, model.batch_size, stat_collector.train_loss,
                                                stat_collector.dev_loss, stat_collector.dev_score))
            if iter_ % print_every == 0:
                perform_no_grad_dataset_iteration(dev, estimator, model, stat_collector)
                if stat_collector.improved_recently() and save_best_models:
                    best_saved_model_path = estimator.save_checkpoint(stat_collector)
            if stat_collector.train_loss < early_stopping_loss:
                early_stopping = True
                break
        print("\n", end='\n', file=sys.stderr)
        train.deallocate()
        estimator.step_schedulers()
    if best_saved_model_path is not None:
        logger.info("Loading the best checkpoint from \"{}\" for evaluation".format(best_saved_model_path))
        model = estimator.load_checkpoint(best_saved_model_path)
        perform_no_grad_dataset_iteration(dev, estimator, model, stat_collector)
        print("Validation Results => Loss: {:.3f}\tScore: {:.3f}".format(
            stat_collector.dev_loss, stat_collector.dev_score))
        perform_no_grad_dataset_iteration(test, estimator, model, stat_collector)
        print("Test Results => Loss: {:.3f}\tScore: {:.3f}".format(
            stat_collector.test_loss, stat_collector.test_score))
