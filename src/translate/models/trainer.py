from tqdm import tqdm
from translate.configs.loader import ConfigLoader
from translate.configs.utils import get_resource_file
from translate.models.RNN.estimator import STSEstimator
from translate.models.RNN.seq2seq import SequenceToSequence
from translate.models.backend.padder import get_padding_batch_loader
from translate.models.backend.utils import device
from translate.readers.constants import ReaderType
from translate.readers.datareader import AbsDatasetReader
from translate.readers.dummydata import DummyDataset


def prepare_dummy_datasets(configs: ConfigLoader):
    train_ = DummyDataset(configs, ReaderType.TRAIN)
    test_ = DummyDataset(configs, ReaderType.TEST)
    dev_ = DummyDataset(configs, ReaderType.DEV)
    vis_ = DummyDataset(configs, ReaderType.VIS)
    return train_, test_, dev_, vis_


def prepare_parallel_dataset(configs: ConfigLoader):
    raise NotImplementedError


def make_model(configs: ConfigLoader, train_dataset: AbsDatasetReader):
    created_model = SequenceToSequence(configs, train_dataset).to(device)
    created_estimator = STSEstimator(configs, created_model)
    return created_model, created_estimator


if __name__ == '__main__':
    opts = ConfigLoader(get_resource_file("default.yaml"))
    dataset_type = opts.get("reader.dataset.type", must_exist=True)
    epochs = opts.get("trainer.optimizer.epochs", must_exist=True)
    if dataset_type == "dummy":
        train, test, dev, vis = prepare_dummy_datasets(opts)
    elif dataset_type == "parallel":
        train, test, dev, vis = prepare_parallel_dataset(opts)
    else:
        raise NotImplementedError
    model, estimator = make_model(opts, train)

    total_train_loss = 0.0
    total_bleu_score = 0.0
    total_train_instances = 0.0

    for epoch in range(epochs):
        iter_ = 0
        itr_handler = tqdm(get_padding_batch_loader(train, model.batch_size),
                           desc="[E {}/{}]-[B {}]-[L {}]-#Batches Processed".format(
                               epoch + 1, epochs, model.batch_size, 0.0), total=len(train)/model.batch_size, ncols=100)
        for input_tensor_batch, target_tensor_batch in itr_handler:
            iter_ += 1
            loss_value, decoded_word_ids = estimator.step(input_tensor_batch, target_tensor_batch)
            total_bleu_score += train.compute_bleu(target_tensor_batch, decoded_word_ids, ref_is_tensor=True)
            total_train_loss += loss_value
            total_train_instances += 1.0
            itr_handler.set_description("[E {}/{}]-[B {}]-[L {:.3f}, S {:.3f}]-#Batches Processed".format(
                epoch + 1, epochs, model.batch_size, total_train_loss/total_train_instances,
                total_bleu_score/total_train_instances))
            # if iter_ % print_every == 0:
            #    continue