from translate.configs.loader import ConfigLoader
from translate.models.RNN.seq2seq import SequenceToSequence
from translate.models.backend.utils import backend


def create_optimizer(optimizer_name, unfiltered_params, lr):
    params = filter(lambda x: x.requires_grad, unfiltered_params)
    if optimizer_name == "adam":
        return backend.optim.Adam(params, lr=lr)
    elif optimizer_name == "adadelta":
        return backend.optim.Adadelta(params, lr=lr)
    elif optimizer_name == "sgd":
        return backend.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError("No optimiser found with the name {}".format(optimizer_name))


class STSEstimator:
    def __init__(self, configs: ConfigLoader, model: SequenceToSequence):
        self.optim_name = configs.get("trainer.optimizer.name", must_exist=True)
        self.learning_rate = configs.get("trainer.optimizer.lr", must_exist=True)
        self.grad_clip_norm = configs.get("trainer.optimizer.gcn", 5)
        self.model = model
        self.encoder_optimizer = create_optimizer(self.optim_name, model.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = create_optimizer(self.optim_name, model.decoder.parameters(), lr=self.learning_rate)

    def step(self, input_tensor_batch, target_tensor_batch):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        batch_loss, batch_loss_size, decoded_word_ids = self.model.forward(input_tensor_batch, target_tensor_batch)
        batch_loss.backward()
        if self.grad_clip_norm > 0.0:
            backend.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.grad_clip_norm)
            backend.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), self.grad_clip_norm)
        loss_value = batch_loss.item() / batch_loss_size
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss_value, decoded_word_ids
