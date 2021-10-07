"""
This file is intended for testing the process regardless of the model accuracy.
 The model should ideally have a BLEU score of 100 in case source and target languages are the same.
"""
import torch
import torch.nn as nn
import torchtext
if torchtext.__version__.startswith('0.9') or torchtext.__version__.startswith('0.10'):
    from torchtext.legacy import data
else:
    from torchtext import data
from models.general import NMTModel
from configuration import device


class CopyModel(NMTModel):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        """
        :param SRC: the trained torchtext.data.Field object containing the source side vocabulary
        :param TGT: the trained torchtext.data.Field object containing the target side vocabulary
        """
        super(CopyModel, self).__init__(SRC, TGT)
        self.dummy_loss = nn.Linear(1, 1, bias=False).to(device)

    def forward(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        """
        tokens_count = 1.0
        cumulative_loss = self.dummy_loss(torch.zeros(1).to(device))
        loss_size = 1.0
        target_length, batch_size = input_tensor_with_lengths[0].size()
        # max_attention_indices = torch.zeros(target_length, batch_size)  # initially
        max_attention_indices = torch.arange(0, target_length).unsqueeze(-1).expand(target_length, batch_size).to(device)  # ideally
        result = input_tensor_with_lengths[0]
        return result, max_attention_indices, cumulative_loss,  loss_size, tokens_count

    def encode(self, input_tensor_with_lengths):
        raise NotImplementedError

    def decode(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False, beam_size=1):
        raise NotImplementedError

    def encoder_init(self, batch_size):
        raise NotImplementedError

    def decoder_init(self, batch_size):
        raise NotImplementedError
